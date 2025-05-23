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
import pandas as pd
from typing import List, Tuple, Literal

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
parser.add_argument('--available_time', type=int, default=2, help='Available time in minutes')
parser.add_argument('--log_output_fields', nargs='*', default=['Ux', 'Uy', 'Exx', 'Eyy', 'Exy', 'Sxx', 'Syy', 'Sxy'], help='Fields to log')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss_fn', nargs='+', default='MSE', help='Loss functions')
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1,1,1,1,1,1,1,1], help='Loss weights (more on DIC points)')
parser.add_argument('--num_point_PDE', type=int, default=10000, help='Number of collocation points for PDE evaluation')
parser.add_argument('--num_point_test', type=int, default=10000, help='Number of test points')

parser.add_argument('--net_width', type=int, default=32, help='Width of the network')
parser.add_argument('--net_depth', type=int, default=5, help='Depth of the network')
parser.add_argument('--activation', choices=['tanh', 'relu', 'elu'], default='tanh', help='Activation function')
parser.add_argument('--optimizer', choices=['adam'], default='adam', help='Optimizer')
parser.add_argument('--mlp', choices=['mlp', 'modified_mlp'], default='mlp', help='Type of MLP for SPINN')
parser.add_argument('--initialization', choices=['Glorot uniform', 'He normal'], default='Glorot uniform', help='Initialization method')

parser.add_argument('--measurments_type', choices=['displacement','strain'], default='strain', help='Type of measurements')
parser.add_argument('--num_measurments', nargs='+', type=int, default=[12,10], help='Number of measurements (should be a perfect square)')
parser.add_argument('--noise_magnitude', type=float, default=1e-6, help='Gaussian noise magnitude (not for DIC simulated)')
parser.add_argument('--u_0', nargs='+', type=float, default=[0,0], help='Displacement scaling factor for Ux and Uy, default(=0) use measurements norm')
parser.add_argument('--s_0', nargs='+', type=float, default=[1,1,1], help='Stress scaling factor for Sxx, Syy, and Sxy')
parser.add_argument('--params_iter_speed', nargs='+', type=float, default=[1,1], help='Scale iteration step for each parameter')
parser.add_argument('--coord_normalization', type=bool, default=True, help='Normalize the input coordinates')
parser.add_argument('--stress_integral', type=bool, default=False, help='Impose stress integral to be equal to the side load')
parser.add_argument('--material_law', choices=['isotropic', 'orthotropic'], default='isotropic', help='Material law')

parser.add_argument('--FEM_dataset', type=str, default='3x3mm.dat', help='Path to FEM data')
parser.add_argument('--DIC_dataset_path', type=str, default='no_dataset', help='If default no_dataset, use FEM model for measurements')
parser.add_argument('--DIC_dataset_number', type=int, default=1, help='Only for DIC simulated measurements')
parser.add_argument('--results_path', type=str, default='results_inverse', help='Path to save results')

args = parser.parse_args()

if len(args.log_output_fields[0]) == 0:
    args.log_output_fields = [] # Empty list for no logging

# For strain measurements, extend loss weights
if args.measurments_type == "strain":
    args.loss_weights.append(args.loss_weights[-1])

dde.config.set_default_autodiff("forward")

# =============================================================================
# 3. Global Constants, Geometry, and Material Parameters
# =============================================================================
# Geometry parameters (mm)
J1S, J1L = 15.0, 23.0    # left clamp lengths
J2S, J2L = 15.0, 23.0    # right clamp lengths
gauge    = 30.0          # central gauge length
w        = 20.0          # specimen height
t        =  2.3          # thickness depth
L     = gauge + J1L + J2L  # total length
u_right = -0.637382 # right clamp displacement from FEM simulation

x_max = (L, w) if args.coord_normalization else (1.0, 1.0) 

# Material parameters (converted to N/mm^2)
if args.material_law.lower() == "isotropic":
    # isotropic plane‐stress
    E       = 52e3     # N/mm^2
    nu      = 0.3
    F_target = -2544.0

    def constitutive_stress(eps_xx, eps_yy, eps_xy):
        # plane‐stress modified constants
        C1 = E/(1 - nu**2)      # factor for normal
        C2 = E/(1 + nu)         # factor for shear
        σ_xx = C1*(eps_xx + nu*eps_yy)
        σ_yy = C1*(eps_yy + nu*eps_xx)
        σ_xy = C2*eps_xy
        return σ_xx, σ_yy, σ_xy

elif args.material_law.lower() == "orthotropic":
    # orthotropic plane‐stress
    Q11, Q22, Q12, Q66 = 41e3, 10.3e3, 3.1e3, 4e3   # N/mm^2
    F_target = -702.0

    def constitutive_stress(eps_xx, eps_yy, eps_xy):
        # plane‐stress modified constants
        σ_xx = Q11*eps_xx + Q12*eps_yy
        σ_yy = Q12*eps_xx + Q22*eps_yy
        σ_xy = 2*Q66*eps_xy
        return σ_xx, σ_yy, σ_xy

# Create trainable scaling factors (one per parameter)
params_factor = [dde.Variable(1 / s) for s in args.params_iter_speed]
trainable_variables = params_factor

# =============================================================================
# 4. Load FEM Data and Build Interpolation Functions
# =============================================================================
dir_path = os.path.dirname(os.path.realpath(__file__))
fem_file = os.path.join(dir_path, r"data_fem", args.FEM_dataset)
n_rows, n_cols = map(int, args.FEM_dataset.replace('.dat', '').split('_')[-1].split('x'))

data = np.loadtxt(fem_file)
x_val      = data[:, 0]
y_val      = data[:, 1]
u_val      = data[:, 2:4]
strain_val = data[:, 4:7]
stress_val = data[:, 7:10]
solution_val = np.hstack((u_val, stress_val))

# Interpolate solution
x_grid = np.unique(x_val)
y_grid = np.unique(y_val)

def create_interpolation_fn(data_array):
    num_components = data_array.shape[1]
    interpolators = []
    for i in range(num_components):
        interp = RegularGridInterpolator(
            (x_grid, y_grid),
            data_array[:, i].reshape(n_rows, n_cols).T,
        )
        interpolators.append(interp)
    def interpolation_fn(x):
        x_in = transform_coords([x[0], x[1]])
        return np.array([interp((x_in[:, 0], x_in[:, 1])) for interp in interpolators]).T
    return interpolation_fn

solution_fn = create_interpolation_fn(solution_val)
strain_fn   = create_interpolation_fn(strain_val)

def strain_from_output(x, f):
    """
    Compute strain components from the network output for strain measurements.
    """
    x = transform_coords(x)
    E_xx = dde.grad.jacobian(f, x, i=0, j=0)[0]
    E_yy = dde.grad.jacobian(f, x, i=1, j=1)[0]
    E_xy = 0.5 * (dde.grad.jacobian(f, x, i=0, j=1)[0] + dde.grad.jacobian(f, x, i=1, j=0)[0])
    return jnp.hstack([E_xx, E_yy, E_xy])

# =============================================================================
# 5.1 Setup Integral Constraint for Stress Integral
# =============================================================================
n_integral = 100
x_integral_top = np.linspace(L-J2L+gauge, L, n_integral)
y_integral_top = np.ones(1) * w

x_integral_bottom = np.linspace(L-J2S, L, n_integral)
y_integral_bottom = np.zeros(1)


integral_points_top = [x_integral_top.reshape(-1, 1), y_integral_top.reshape(-1, 1)]
integral_points_bottom = [x_integral_bottom.reshape(-1, 1), y_integral_bottom.reshape(-1, 1)]

def integral_stress(x, outputs, X):
    x = transform_coords(x)
    x_grid = x[:, 0]
    Sxx = outputs[0][:, 2]
    return jnp.trapezoid(Sxx, x_grid, axis=0)

# Integral_BC = dde.PointSetOperatorBC(integral_points, (b+m*L_max/2) * L_max, integral_stress)
# bcs = [Integral_BC]

# TODO: need to take the 2 integral points into on BC to compute the full integral

# =============================================================================
# 5.2 Setup Measurement Data Based on Type (Displacement, Strain, DIC)
# =============================================================================
if args.measurments_type == "displacement":
    if args.DIC_dataset_path != "no_dataset":
        dic_path = os.path.join(dir_path, args.DIC_dataset_path)
        dic_number = args.DIC_dataset_number
        X_dic = pd.read_csv(os.path.join(dic_path, "x", f"x_{dic_number}.csv"), delimiter=";").dropna(axis=1).to_numpy()
        Y_dic = pd.read_csv(os.path.join(dic_path, "y", f"y_{dic_number}.csv"), delimiter=";").dropna(axis=1).to_numpy()
        Ux_dic = pd.read_csv(os.path.join(dic_path, "ux", f"ux_{dic_number}.csv"), delimiter=";").dropna(axis=1).to_numpy().T.reshape(-1, 1)
        Uy_dic = pd.read_csv(os.path.join(dic_path, "uy", f"uy_{dic_number}.csv"), delimiter=";").dropna(axis=1).to_numpy().T.reshape(-1, 1)
        DIC_data = np.hstack([Ux_dic, Uy_dic])
        x_values = np.mean(X_dic, axis=0).reshape(-1, 1)
        y_values = np.mean(Y_dic, axis=1).reshape(-1, 1)
        X_DIC_input = [x_values, y_values]
        if args.num_measurments != x_values.shape[0] * y_values.shape[0]:
            print(f"For this DIC dataset, the number of measurements is fixed to {x_values.shape[0] * y_values.shape[0]}")
            args.num_measurments = x_values.shape[0] * y_values.shape[0]
    else:
        X_DIC_input = [np.linspace(J1L, J1L+gauge, args.num_measurments[0]).reshape(-1, 1),
                      np.linspace(0, w, args.num_measurments[1]).reshape(-1, 1)]
        DIC_data = solution_fn(X_DIC_input)[:, :2]
        DIC_data += np.random.normal(0, args.noise_magnitude, DIC_data.shape)

    DIC_norms = np.mean(np.abs(DIC_data), axis=0) # to normalize the loss
    measure_Ux = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 0:1]/DIC_norms[0],
                                          lambda x, f, x_np: f[0][:, 0:1]/DIC_norms[0])
    measure_Uy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 1:2]/DIC_norms[1],
                                          lambda x, f, x_np: f[0][:, 1:2]/DIC_norms[1])
    bcs = [measure_Ux, measure_Uy]

elif args.measurments_type == "strain":
    if args.DIC_dataset_path != "no_dataset":
        dic_path = os.path.join(dir_path, args.DIC_dataset_path)
        dic_number = args.DIC_dataset_number
        X_dic = pd.read_csv(os.path.join(dic_path, "x", f"x_{dic_number}.csv"), delimiter=";").dropna(axis=1).to_numpy()
        Y_dic = pd.read_csv(os.path.join(dic_path, "y", f"y_{dic_number}.csv"), delimiter=";").dropna(axis=1).to_numpy()
        Ux_dic = pd.read_csv(os.path.join(dic_path, "ux", f"ux_{dic_number}.csv"), delimiter=";").dropna(axis=1).to_numpy().T.reshape(-1, 1)
        Uy_dic = pd.read_csv(os.path.join(dic_path, "uy", f"uy_{dic_number}.csv"), delimiter=";").dropna(axis=1).to_numpy().T.reshape(-1, 1)
        E_xx_dic = pd.read_csv(os.path.join(dic_path, "exx", f"exx_{dic_number}.csv"), delimiter=";").dropna(axis=1).to_numpy().T.reshape(-1, 1)
        E_yy_dic = pd.read_csv(os.path.join(dic_path, "eyy", f"eyy_{dic_number}.csv"), delimiter=";").dropna(axis=1).to_numpy().T.reshape(-1, 1)
        E_xy_dic = pd.read_csv(os.path.join(dic_path, "exy", f"exy_{dic_number}.csv"), delimiter=";").dropna(axis=1).to_numpy().T.reshape(-1, 1)
        x_values = np.mean(X_dic, axis=0).reshape(-1, 1)
        y_values = np.mean(Y_dic, axis=1).reshape(-1, 1)
        X_DIC_input = [x_values, y_values]
        DIC_data = np.hstack([E_xx_dic, E_yy_dic, E_xy_dic])
        if args.num_measurments != x_values.shape[0] * y_values.shape[0]:
            print(f"For this DIC dataset, the number of measurements is fixed to {x_values.shape[0] * y_values.shape[0]}")
            args.num_measurments = x_values.shape[0] * y_values.shape[0]
    else:
        X_DIC_input = [np.linspace(J1L, J1L+gauge, args.num_measurments[0]).reshape(-1, 1),
                      np.linspace(0, w, args.num_measurments[1]).reshape(-1, 1)]
        DIC_data = strain_fn(X_DIC_input)
        DIC_data += np.random.normal(0, args.noise_magnitude, DIC_data.shape)
    DIC_norms = np.mean(np.abs(DIC_data), axis=0) # to normalize the loss
    measure_Exx = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 0:1]/DIC_norms[0],
                                           lambda x, f, x_np: strain_from_output(x, f)[:, 0:1]/DIC_norms[0])
    measure_Eyy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 1:2]/DIC_norms[1],
                                           lambda x, f, x_np: strain_from_output(x, f)[:, 1:2]/DIC_norms[1])
    measure_Exy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 2:3]/DIC_norms[2],
                                           lambda x, f, x_np: strain_from_output(x, f)[:, 2:3]/DIC_norms[2])
    bcs = [measure_Exx, measure_Eyy, measure_Exy]

else:
    raise ValueError("Invalid measurement type. Choose 'displacement' or 'strain'.")

# Use measurements norm as the default scaling factor
if args.DIC_dataset_path != "no_dataset":
    disp_norms = np.mean(np.abs(np.hstack([Ux_dic, Uy_dic])), axis=0)
else:
    disp_norms = np.mean(np.abs(solution_fn(X_DIC_input)[:, :2]), axis=0)
args.u_0 = [disp_norms[i] if not args.u_0[i] else args.u_0[i] for i in range(2)]

# =============================================================================
# 6. PINN Implementation: Boundary Conditions and PDE Residual
# =============================================================================
# Define the domain geometry
geom = dde.geometry.Rectangle([0, 0], [L, w])

def bc_factor(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    segments: List[Tuple[Tuple[float,float], Tuple[float,float]]],
    smoothness: Literal["C0", "C0+"] = "C0",
) -> jnp.ndarray:
    """
    x1, x2 : (n,) arrays of coordinates
    segments: list of ((xA,yA),(xB,yB))
    smoothness: "C0" = min distance, "C0+" = product of distances
    returns: (n,1) array
    """
    # helper: distance from (x1,x2) to segment A→B
    def _dist(A, B):
        xA, yA = A; xB, yB = B
        vx, vy = xB - xA, yB - yA
        # vector A→P
        px = x1 - xA
        py = x2 - yA
        t = jnp.clip((px*vx + py*vy) / (vx*vx + vy*vy), 0.0, 1.0)
        qx = xA + t*vx
        qy = yA + t*vy
        return jnp.hypot(x1 - qx, x2 - qy)[:, None]

    # build (n, m) matrix of distances
    D = jnp.hstack([ _dist(A, B) for A, B in segments ])

    raw = jnp.min(D, axis=1, keepdims=True)    if smoothness=="C0" else jnp.prod(D, axis=1, keepdims=True)
    M = raw.max()
    M = jnp.where(M > 0, M, 1.0)    # avoid dividing by 0
    return (raw / M).flatten()

# def Uy_bc(x1, x2, J1L, J1S, J2L, J2S, L, w, U,
#           coord_normalization=False):
#     if coord_normalization:
#         J1L, J1S, J2L, J2S, L = [v/L for v in (J1L, J1S, J2L, J2S, L)]
#         w = w / w
#     t     = x2 / w
#     start = J1L + (J1S - J1L) * t
#     end   = (L - J2S) + ((L - J2L) - (L - J2S)) * t
#     eps = 1e-6
#     ramp_den = end - start
#     Uy = jnp.where(
#         x1 >= end, U,
#         jnp.where(x1 > start,
#                 (x1 - start) / (ramp_den + eps) * U,
#                 0.0))
#     return Uy

def smoothstep(x):
    """Smoothstep function: 0 if x<0, 1 if x>1, smooth in between."""
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def Uy_bc(x1, x2, J1L, J1S, J2L, J2S, L, w, U,
    coord_normalization=False):
    if coord_normalization:
        J1L, J1S, J2L, J2S, L = [v / L for v in (J1L, J1S, J2L, J2S, L)]
        w = w / w # this line simplifies to 1.0
    t = x2 / w
    start = J1L + (J1S - J1L) * t
    end = (L - J2S) + ((L - J2L) - (L - J2S)) * t
    ramp = smoothstep((x1 - start) / (end - start + 1e-8))
    return ramp * U

contact_eps = 1e-2 * L

segs_uy = [
    ((0.0, 0.0), (J1L - contact_eps, 0.0)), # left bottom 
    ((0.0, w), (J1S - contact_eps, w)), # left top
    ((L-J2S + contact_eps, 0.0), (L, 0.0)), # right bottom
    ((L-J2L + contact_eps, w), (L, w)), # right top
]

segs_Sxx = [
    ((0.0, 0.0), (0.0, w)), # left 
    ((L, 0.0), (L, w)), # right
]

segs_Syy = [
    ((J1L + contact_eps, 0.0), (L-J2S - contact_eps, 0.0)), # bottom gauge
    ((J1S + contact_eps, w), (L-J2L - contact_eps, w)), # top gauge
]

segs_Sxy = [
    ((0.0, 0.0), (J1L - contact_eps, 0.0)), # bottom left
    ((J1L + contact_eps, 0.0), (L-J2S - contact_eps, 0.0)), # bottom gauge
    ((L-J2S + contact_eps, 0.0), (L, 0.0)), # bottom right
    ((0.0, w), (J1S - contact_eps, w)), # top left
    ((J1S + contact_eps, w), (L-J2L - contact_eps, w)), # top gauge
    ((L-J2L + contact_eps, w), (L, w)), # top right
    ((0.0, 0.0), (0.0, w)), # left
    ((L, 0.0), (L, w)), # right
]

all_segs = [segs_uy, segs_Sxx, segs_Syy, segs_Sxy]
if args.coord_normalization:
    for segs in all_segs:
        for i, seg in enumerate(segs):
            segs[i] = [(v[0]/L, v[1]/w) for v in seg]


def HardBC(x, f, all_segs=all_segs):
    """
    Apply hard boundary conditions via transformation.
    If x is provided as a list of 1D arrays, transform it to a 2D meshgrid.
    """
    if isinstance(x, list):
        x = transform_coords(x)
    
    segs_uy, segs_Sxx, segs_Syy, segs_Sxy = all_segs
    Ux  = f[:, 0] * args.u_0[0] 
    Uy  = f[:, 1] * args.u_0[1] * bc_factor(x[:, 0], x[:, 1], segs_uy, "C0+") + Uy_bc(x[:, 0], x[:, 1], J1L, J1S, J2L, J2S, L, w, U=u_right, coord_normalization=args.coord_normalization)
    Sxx = f[:, 2] * args.s_0[0] * bc_factor(x[:, 0], x[:, 1], segs_Sxx, "C0+")
    Syy = f[:, 3] * args.s_0[1] * bc_factor(x[:, 0], x[:, 1], segs_Syy, "C0+") 
    Sxy = f[:, 4] * args.s_0[2] * bc_factor(x[:, 0], x[:, 1], segs_Sxy, "C0+")
    return dde.backend.stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)

def pde(x, f, unknowns=params_factor):
    """
    Define the PDE residuals for the linear elastic plate.
    """
    x = transform_coords(x)
    
    E_xx = dde.grad.jacobian(f, x, i=0, j=0)[0]
    E_yy = dde.grad.jacobian(f, x, i=1, j=1)[0]
    E_xy = 0.5 * (dde.grad.jacobian(f, x, i=0, j=1)[0] + dde.grad.jacobian(f, x, i=1, j=0)[0])
    
    S_xx, S_yy, S_xy = constitutive_stress(E_xx, E_yy, E_xy)

    Sxx_x = dde.grad.jacobian(f, x, i=2, j=0)[0]
    Syy_y = dde.grad.jacobian(f, x, i=3, j=1)[0]
    Sxy_x = dde.grad.jacobian(f, x, i=4, j=0)[0]
    Sxy_y = dde.grad.jacobian(f, x, i=4, j=1)[0]
    
    momentum_x = Sxx_x + Sxy_y
    momentum_y = Sxy_x + Syy_y
    
    f_val = f[0] # f[1] is the function 
    stress_x  = S_xx - f_val[:, 2:3]
    stress_y  = S_yy - f_val[:, 3:4]
    stress_xy = S_xy - f_val[:, 4:5]
    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]

def input_scaling(x):
    """
    Scale the input coordinates to the range [0, 1].
    """
    if isinstance(x, list):
        return [x[0] / x_max[0], x[1] / x_max[1]]
    else:
        return x / np.array(x_max) # TODO: check if this is correct
# =============================================================================
# 7. Define Neural Network, Data, and Model
# =============================================================================
layers = [2] + [args.net_width] * args.net_depth + [5]
net = dde.nn.SPINN(layers, args.activation, args.initialization, args.mlp)
batch_size = args.num_point_PDE + args.num_measurments[0] * args.num_measurments[1]
num_params = sum(p.size for p in jax.tree.leaves(net.init(jax.random.PRNGKey(0), jnp.ones(layers[0]))))

data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=args.num_point_PDE,
    solution=solution_fn,
    num_test=args.num_point_test,
    is_SPINN=True,
)
if args.coord_normalization:
    net.apply_feature_transform(input_scaling)
net.apply_output_transform(HardBC)

model = dde.Model(data, net)
model.compile(args.optimizer, lr=args.lr, metrics=["l2 relative error"],
              loss_weights=[1]*len(args.loss_weights), loss=args.loss_fn,
              external_trainable_variables=trainable_variables)

# =============================================================================
# 8. Setup Callbacks for Logging
# =============================================================================
results_path = os.path.join(dir_path, args.results_path)
if args.DIC_dataset_path != "no_dataset":
    dic_prefix = 'dic_'
    noise_prefix = args.DIC_dataset_path.split('/')[-1]
else:
    dic_prefix = ''
    noise_prefix = f"{args.noise_magnitude}noise"
num_measurments_str = f"{X_DIC_input[0].shape[0]}x{X_DIC_input[1].shape[0]}"
folder_name = f"{dic_prefix}{args.measurments_type}_{num_measurments_str}_{noise_prefix}_{args.available_time if args.available_time else args.n_iter}{'min' if args.available_time else 'iter'}"
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
callbacks.append(dde.callbacks.VariableValue(params_factor, period=args.log_every,
                                               filename=os.path.join(new_folder_path, "variables_history.dat"),
                                               precision=8))

# Log the history of the output fields
def output_log(x, output, field):
    if field in ['Ux', 'Uy', 'Sxx', 'Syy', 'Sxy']:
        return output[0][:, ['Ux', 'Uy', 'Sxx', 'Syy', 'Sxy'].index(field)]
    if field in ['Exx', 'Eyy', 'Exy']:
        return strain_from_output(x, output)[:, ['Exx', 'Eyy', 'Exy'].index(field)]
    raise ValueError(f"Invalid field name: {field}")
        
X_plot = [np.linspace(0, L, 150).reshape(-1, 1), np.linspace(0, w, 40).reshape(-1, 1)]
for i, field in enumerate(args.log_output_fields): # Log output fields
    callbacks.append(
        dde.callbacks.OperatorPredictor(
            X_plot,
            lambda x, output, field=field: output_log(x, output, field),
            period=args.log_every,
            filename=os.path.join(new_folder_path, f"{field}_history.dat"),
            precision=6
        )
    )

# =============================================================================
# 9. Calculate Loss Weights based on the Gradient of the Loss Function
# =============================================================================
from jax.flatten_util import ravel_pytree
def loss_function(params,comp=0,inputs=[X_DIC_input]*len(bcs)+[X_plot]):
    return model.outputs_losses_train(params, inputs, None)[1][comp]

n_loss = len(args.loss_weights)

def calc_loss_weights(model):
    loss_grads = [1]*n_loss

    for i in range(n_loss):
        grad_fn = jax.grad(lambda params,comp=i: loss_function(params,comp))
        grads = grad_fn(model.params)[0]
        flattened_grad = ravel_pytree(list(grads.values())[0])[0]
        loss_grads[i] = jnp.linalg.norm(flattened_grad)

    loss_grads = jnp.array(loss_grads)
    loss_weights_grads = jnp.sqrt(jnp.sum(loss_grads)/loss_grads) # Caution: ad-hoc sqrt
    return loss_weights_grads, loss_grads

loss_weights_grads, loss_grads = calc_loss_weights(model)
new_loss_weights = [w * g for w, g in zip(args.loss_weights, loss_weights_grads)]
model.compile(args.optimizer, lr=args.lr, metrics=["l2 relative error"],
              loss_weights=new_loss_weights, loss=args.loss_fn,
              external_trainable_variables=trainable_variables)

# =============================================================================
# 10. Training
# =============================================================================
start_time = time.time()
# print(f"E(GPa): {E_init * params_factor[0].value * args.params_iter_speed[0]/1e3:.3f}, nu: {nu_init * params_factor[1].value * args.params_iter_speed[1]:.3f}")
losshistory, train_state = model.train(iterations=args.n_iter, callbacks=callbacks, display_every=args.log_every)
elapsed = time.time() - start_time

# =============================================================================
# 11. Logging
# =============================================================================
dde.utils.save_loss_history(losshistory, os.path.join(new_folder_path, "loss_history.dat"))

# params_init = [E_init, nu_init]
variables_history_path = os.path.join(new_folder_path, "variables_history.dat")

# Read the variables history
# with open(variables_history_path, "r") as f:
#     lines = f.readlines()

# Update the variables history with scaled values
# with open(variables_history_path, "w") as f:
#     for line in lines:
#         step, value = line.strip().split(' ', 1)
#         values = [scale * init * val for scale, init, val in zip(args.params_iter_speed, params_init, eval(value))]
#         f.write(f"{step} " + dde.utils.list_to_str(values, precision=8) + "\n")

# Read the variables history
# with open(variables_history_path, "r") as f:
#     lines = f.readlines()
# Final E and nu values as the average of the last 10 values 
# E_final = np.mean([eval(line.strip().split(' ', 1)[1])[0] for line in lines[-10:]])
# nu_final = np.mean([eval(line.strip().split(' ', 1)[1])[1] for line in lines[-10:]])
# print(f"Final E(GPa): {E_final/1e3:.3f}, nu: {nu_final:.3f}")

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
        "params_iter_speed": args.params_iter_speed,
        "u_0": args.u_0,
        "logged_fields": args.log_output_fields,
    }
    problem_info = {
        "L": L,
        "w": w,
        "t": t,
        "J1S": J1S,
        "J1L": J1L,
        "J2S": J2S,
        "J2L": J2L,
        "gauge": gauge,
        "u_right": u_right,
        # "E_actual": E_actual,
        # "nu_actual": nu_actual,
        # "E_init": E_init,
        # "nu_init": nu_init,
        # "E_final": E_final,
        # "nu_final": nu_final,
    }
    data_info = {
        "num_measurments": args.num_measurments,
        "noise_magnitude": args.noise_magnitude,
        "measurments_type": args.measurments_type,
        "DIC_dataset_path": args.DIC_dataset_path,
        "DIC_dataset_number": args.DIC_dataset_number,
        "FEM_dataset": args.FEM_dataset, 
    }
    info = {"system": system_info, "gpu": gpu_info, "execution": execution_info,
            "network": network_info, "problem": problem_info, "data": data_info}
    with open(fname, "w") as f:
        json.dump(info, f, indent=4)

log_config(os.path.join(new_folder_path, "config.json"))