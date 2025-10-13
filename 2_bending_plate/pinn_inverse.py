"""
PINN-MT2.0: Inverse identification for the side loaded plate example

This script replicates the example from Martin et al. (2019) using a FEM reference
solution and a Physics-Informed Neural Network (PINN) to identify material properties.
"""

import os
import time
import json
import argparse
import platform
import subprocess

import numpy as np
import jax
import jax.numpy as jnp
import deepxde as dde
from scipy.interpolate import RegularGridInterpolator

# =============================================================================
# 1. Utility Function: Coordinate Transformation for SPINN
# =============================================================================
def transform_coords(x):
    """
    For SPINN, if the input x is provided as a list of 1D arrays (e.g., [X_coords, Y_coords]),
    this function creates a 2D meshgrid and stacks the results into a 2D coordinate array.
    """
    x_mesh = [x_.ravel() for x_ in jnp.meshgrid(jnp.atleast_1d(x[0].squeeze()), jnp.atleast_1d(x[1].squeeze()), indexing="ij")]
    return dde.backend.stack(x_mesh, axis=-1)

# =============================================================================
# 2. Parse Arguments
# =============================================================================
parser = argparse.ArgumentParser(description="Physics Informed Neural Networks for Linear Elastic Plate")
parser.add_argument('--n_iter', type=int, default=int(1e10), help='Number of iterations')
parser.add_argument('--log_every', type=int, default=250, help='Log every n steps')
parser.add_argument('--available_time', type=int, default=5, help='Available time in minutes (overrides n_iter except if 0)')
parser.add_argument('--log_output_fields', nargs='*', default=['W'], help='Fields to log')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss_fn', nargs='+', default='MSE', help='Loss functions')
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1,1e4,1e4,1e4,1e4,1e1], help='Loss weights (more on DIC points)')
parser.add_argument('--num_point_PDE', type=int, default=1000, help='Number of collocation points for PDE evaluation')
parser.add_argument('--num_point_test', type=int, default=10000, help='Number of test points')

parser.add_argument('--net_width', type=int, default=32, help='Width of the network')
parser.add_argument('--net_depth', type=int, default=2, help='Depth of the network')
parser.add_argument('--net_rank', type=int, default=32, help='Rank of the network')
parser.add_argument('--activation', choices=['tanh', 'relu', 'elu'], default='tanh', help='Activation function')
parser.add_argument('--optimizer', choices=['adam'], default='adam', help='Optimizer')
parser.add_argument('--mlp', choices=['mlp', 'modified_mlp'], default='mlp', help='Type of MLP for SPINN')
parser.add_argument('--initialization', choices=['Glorot uniform', 'He normal'], default='Glorot uniform', help='Initialization method')

parser.add_argument('--measurments_type', choices=['displacement','DIC'], default='displacement', help='Type of measurements')
parser.add_argument('--n_measurments', type=int, default=100, help='Number of measurements (should be a perfect square)')
parser.add_argument('--noise_magnitude', type=float, default=1e-6, help='Gaussian noise magnitude')
parser.add_argument('--w_0', nargs='+', type=float, default=0, help='Displacement scaling factor for Ux and Uy, default(=0) use measurements norm')
parser.add_argument('--param_iter_speed', nargs='+', type=float, default=1, help='Scale iteration step for each parameter')

parser.add_argument('--FEM_dataset', type=str, default='100x100.dat', help='Path to FEM data')
parser.add_argument('--DIC_dataset', type=str, default='3mm_0noise', help='Only for DIC measurements type')
parser.add_argument('--results_path', type=str, default='results', help='Path to save results')

args = parser.parse_args()

if len(args.log_output_fields[0]) == 0:
    args.log_output_fields = [] # Empty list for no logging

dde.config.set_default_autodiff("forward")

# =============================================================================
# 3. Global Constants, Geometry, and Material Parameters
# =============================================================================
L_max = 1.0  # Length of the square plate
E_actual = 1e9  # Young's modulus (Pa)
nu_actual = 0.3  # Poisson's ratio
t = 1e-3  # Plate thickness (m)
q = 1  # Uniform load (Pa)
w0 = 1e-2 # Order of magnitude of the displacement

E_init = 0.6e9 # Initial guess for Young's modulus
nu_init = 0.2  # Initial guess for Poisson's ratio

D_init = E_init * t**3 / (12 * (1 - nu_init**2))  # Initial flexural rigidity
D_actual = E_actual * t**3 / (12 * (1 - nu_actual**2))  # Actual flexural rigidity

# Create trainable scaling factors (one per parameter)
param_factor = dde.Variable(1 / args.param_iter_speed) 
trainable_variables = param_factor

# =============================================================================
# 4. Load FEM Data and Build Interpolation Functions
# =============================================================================
dir_path = os.path.dirname(os.path.realpath(__file__))
fem_file = os.path.join(dir_path, r"data_fem", args.FEM_dataset)
data = np.loadtxt(fem_file)
X_val = data[:, :2]
w_val = data[:, 2:3]
solution_val = w_val

n_mesh_points = int(np.sqrt(X_val.shape[0]))
x_grid = np.linspace(0, L_max, n_mesh_points)
y_grid = np.linspace(0, L_max, n_mesh_points)

interp = RegularGridInterpolator((x_grid, y_grid), solution_val.reshape(n_mesh_points, n_mesh_points).T)
def solution_fn(x):
    x_in = transform_coords([x[0], x[1]])
    return interp(x_in).reshape(-1, 1)

# =============================================================================
# 5. Setup Measurement Data Based on Type (Displacement, Strain, DIC)
# =============================================================================
args.n_measurments = int(np.sqrt(args.n_measurments))**2
if args.measurments_type == "displacement":    
    X_DIC_input = [np.linspace(0, L_max, args.n_measurments).reshape(-1, 1)] * 2
    DIC_data = solution_fn(X_DIC_input)[:, :2]
    DIC_data += np.random.normal(0, args.noise_magnitude, DIC_data.shape)
    DIC_norm = np.mean(np.abs(DIC_data), axis=0) # to normalize the loss
    measure_W = dde.PointSetOperatorBC(X_DIC_input, DIC_data/DIC_norm,
                                          lambda x, f, x_np: f[0]/DIC_norm)
    bcs = [measure_W]

elif args.measurments_type == "DIC":
    import pandas as pd
    dic_path = os.path.join(dir_path, f"dic_data/{args.DIC_dataset}")
    X_dic = pd.read_csv(os.path.join(dic_path, "X.csv"),
                        delimiter=";").dropna(axis=1).to_numpy()
    Y_dic = pd.read_csv(os.path.join(dic_path, "Y.csv"),
                        delimiter=";").dropna(axis=1).to_numpy()
    W_dic = pd.read_csv(os.path.join(dic_path, "W.csv"),
                         delimiter=";").dropna(axis=1).to_numpy().T.reshape(-1, 1)
    x_values = np.mean(X_dic, axis=0).reshape(-1, 1)
    y_values = np.mean(Y_dic, axis=1).reshape(-1, 1)
    X_DIC_input = [x_values, y_values]
    
    if args.n_measurments != x_values.shape[0] * y_values.shape[0]:
        print(f"For this DIC dataset, the number of measurements is fixed to {x_values.shape[0] * y_values.shape[0]}")
    args.n_measurments = x_values.shape[0] * y_values.shape[0]

    DIC_norm = np.mean(np.abs(W_dic)) # to normalize the loss
    measure_W = dde.PointSetOperatorBC(X_DIC_input, W_dic/DIC_norm,
                                          lambda x, f, x_np: f[0][:, 0:1]/DIC_norm)

    bcs = [measure_W]

# Use measurements norm as the default scaling factor
args.w_0 = DIC_norm.item() if not args.w_0 else args.w_0

# =============================================================================
# 6. PINN Implementation: Boundary Conditions and PDE Residual
# =============================================================================
# Define the domain geometry
geom = dde.geometry.Rectangle([0, 0], [L_max, L_max])

def HardBC(x, f, x_max=L_max):
    """
    Apply hard boundary conditions via transformation.
    If x is provided as a list of 1D arrays, transform it to a 2D meshgrid.
    """
    if isinstance(x, list):
        x = transform_coords(x)
    w = f[:, 0:1]*x[:, 0:1]*(1 - x[:, 0:1])*x[:, 1:2]*(1 - x[:, 1:2])*args.w_0
    return w

def pde(x, f, unknowns=param_factor):
    """
    Define the PDE residuals for the linear elastic plate.
    """
    x = transform_coords(x)
    
    param_factor = unknowns[0]*args.param_iter_speed
    D = D_init * param_factor**2
    # Compute the required derivatives
    w_x1x1 = dde.grad.hessian(f, x, i=0, j=0)  # ∂²w/∂x₁²
    w_x2x2 = dde.grad.hessian(f, x, i=1, j=1)  # ∂²w/∂x₂²
    w_x1x1x1x1 = dde.grad.hessian(w_x1x1, x, i=0, j=0)[0]  # ∂⁴w/∂x₁⁴
    w_x2x2x2x2 = dde.grad.hessian(w_x2x2, x, i=1, j=1)[0]  # ∂⁴w/∂x₂⁴
    w_x1x2x1x2 = dde.grad.hessian(w_x1x1, x, i=1, j=1)[0]  # ∂⁴w/∂x₁²∂x₂²
    
    return w_x1x1x1x1 + 2 * w_x1x2x1x2 + w_x2x2x2x2 - q / D 

bc_point_left = [np.array([0]).reshape(-1,1), np.linspace(0, 1, 100).reshape(-1,1)]
bc_point_right = [np.array([1]).reshape(-1,1), np.linspace(0, 1, 100).reshape(-1,1)]
bc_point_bottom = [np.linspace(0, 1, 100).reshape(-1,1), np.array([0]).reshape(-1,1)]
bc_point_top = [np.linspace(0, 1, 100).reshape(-1,1), np.array([1]).reshape(-1,1)]

def jacobian_spinn(x, f, x_np, j):
    x_in = transform_coords(x)
    return dde.grad.jacobian(f, x_in, i=0, j=j)[0]

bc_left = dde.PointSetOperatorBC(bc_point_left, 0, lambda x, f, x_np, j=0: jacobian_spinn(x, f, x_np, j))
bc_right = dde.PointSetOperatorBC(bc_point_right, 0, lambda x, f, x_np, j=0: jacobian_spinn(x, f, x_np, j))
bc_bottom = dde.PointSetOperatorBC(bc_point_bottom, 0, lambda x, f, x_np, j=1: jacobian_spinn(x, f, x_np, j))
bc_top = dde.PointSetOperatorBC(bc_point_top, 0, lambda x, f, x_np, j=1: jacobian_spinn(x, f, x_np, j))

bcs = [bc_left, bc_right, bc_bottom, bc_top] + bcs

# =============================================================================
# 7. Define Neural Network, Data, and Model
# =============================================================================
layers = [2] + [args.net_width] * args.net_depth + [args.net_rank] + [1]
net = dde.nn.SPINN(layers, args.activation, args.initialization, args.mlp)
batch_size = args.num_point_PDE + args.n_measurments
num_params = sum(p.size for p in jax.tree.leaves(net.init(jax.random.PRNGKey(0), jnp.ones(layers[0]))))

data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=args.num_point_PDE,
    solution=solution_fn,
    num_test= args.num_point_test,
    is_SPINN=True,
)
net.apply_output_transform(HardBC)

model = dde.Model(data, net)
model.compile(args.optimizer, lr=args.lr, metrics=["l2 relative error"],
              loss_weights=args.loss_weights, loss=args.loss_fn,
              external_trainable_variables=trainable_variables)

# =============================================================================
# 8. Setup Callbacks for Logging
# =============================================================================
results_path = os.path.join(dir_path, args.results_path)
folder_name = f"{args.measurments_type}_x{args.n_measurments}_{args.noise_magnitude}noise_{args.available_time if args.available_time else args.n_iter}{'min' if args.available_time else 'iter'}"
existing_folders = [f for f in os.listdir(results_path) if f.startswith(folder_name)]
if existing_folders:
    suffixes = [int(f.split("-")[-1]) for f in existing_folders if f != folder_name]
    folder_name = f"{folder_name}-{max(suffixes)+1}" if suffixes else f"{folder_name}-1"
new_folder_path = os.path.join(results_path, folder_name)
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

callbacks = []
if args.available_time:
    callbacks.append(dde.callbacks.Timer(args.available_time))
callbacks.append(dde.callbacks.VariableValue(param_factor, period=args.log_every,
                                               filename=os.path.join(new_folder_path, "variables_history.dat"),
                                               precision=8))
X_plot = [np.linspace(0, L_max, 100).reshape(-1, 1)] * 2
for i, field in enumerate(args.log_output_fields): # Log output fields
    callbacks.append(
        dde.callbacks.OperatorPredictor(
            X_plot,
            lambda x, output, i=i: output[0][:, i],
            period=args.log_every,
            filename=os.path.join(new_folder_path, f"{field}_history.dat"),
            precision=6
        )
    )

# =============================================================================
# 9. Training
# =============================================================================
start_time = time.time()
print(f"Initial D: {D_init*(param_factor.value*args.param_iter_speed)**2:.3e} | {D_actual:.3e}")
losshistory, train_state = model.train(iterations=args.n_iter, callbacks=callbacks, display_every=args.log_every)
elapsed = time.time() - start_time

# =============================================================================
# 10. Logging
# =============================================================================
dde.utils.save_loss_history(losshistory, os.path.join(new_folder_path, "loss_history.dat"))

params_init = [E_init, nu_init]
variables_history_path = os.path.join(new_folder_path, "variables_history.dat")

# Read the variables history
with open(variables_history_path, "r") as f:
    lines = f.readlines()

# Update the variables history with scaled values
with open(variables_history_path, "w") as f:
    for line in lines:
        step, value = line.strip().split(' ', 1)
        value = [D_init*(args.param_iter_speed*eval(value)[0])**2]
        f.write(f"{step} "+dde.utils.list_to_str(value, precision=8)+"\n")

# Read the variables history
with open(variables_history_path, "r") as f:
    lines = f.readlines()
# Final D value as the average of the last 10 values 
D_final = np.mean([eval(line.strip().split(' ', 1)[1])[0] for line in lines[-10:]])
print(f"Final D: {D_final:.3e} | {D_actual:.3e}")

def log_config(fname):
    """
    Save configuration and execution details to a JSON file, grouped by category.
    """
    system_info = {"OS": platform.system(), "Release": platform.release()}
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        gpu_name, total_memory_mb = output.stdout.strip().split(", ")
        total_memory_gb = round(float(total_memory_mb.split(' ')[0]) / 1024, 2)
        gpu_info = {"GPU": gpu_name, "Total GPU Memory": f"{total_memory_gb:.2f} GB"}
    except subprocess.CalledProcessError:
        gpu_info = {"GPU": "No GPU found", "Total GPU Memory": "N/A"}
    
    execution_info = {
        "n_iter": train_state.epoch,
        "elapsed": elapsed,
        "iter_per_sec": train_state.epoch / elapsed,
        "backend": dde.backend.backend_name,
    }
    network_info = {
        "net_width": args.net_width,
        "net_depth": args.net_depth,
        "num_params": num_params,
        "activation": args.activation,
        "mlp_type": args.mlp,
        "optimizer": args.optimizer,
        "initializer": args.initialization,
        "batch_size": batch_size,
        "lr": args.lr,
        "loss_weights": args.loss_weights,
        "param_iter_speed": args.param_iter_speed,
        "u_0": args.w_0,
        "logged_fields": args.log_output_fields,
    }
    problem_info = {
        "L_max": L_max,
        "E_actual": E_actual,
        "nu_actual": nu_actual,
        "E_init": E_init,
        "nu_init": nu_init,
        "D_init": D_init,
        "D_final": D_final,
        "D_actual": D_actual,
        }
    data_info = {
        "n_measurments": (int(np.sqrt(args.n_measurments)))**2,
        "noise_magnitude": args.noise_magnitude,
        "measurments_type": args.measurments_type,
    }
    info = {"system": system_info, "gpu": gpu_info, "execution": execution_info,
            "network": network_info, "problem": problem_info, "data": data_info}
    with open(fname, "w") as f:
        json.dump(info, f, indent=4)

log_config(os.path.join(new_folder_path, "config.json"))