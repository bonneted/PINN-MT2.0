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
import warnings

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
@dde.utils.list_handler
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
parser.add_argument('--n_iter', nargs='+', type=int, default=int(1e10), help='Number of iterations')
parser.add_argument('--log_every', type=int, default=250, help='Log every n steps')
parser.add_argument('--available_time', type=int, default=2, help='Available time in minutes')
parser.add_argument('--log_output_fields', nargs='*', default=['Ux', 'Uy', 'Exx', 'Eyy', 'Exy', 'Sxx', 'Syy', 'Sxy'], help='Fields to log')
parser.add_argument('--lr', nargs='+', type=float, default=0.001, help='Learning rate')
parser.add_argument('--n_retrain', type=int, default=5, help='Number of retraining iterations')
parser.add_argument('--max_adaptive_samples', type=int, default=3, help='Maximum number of adaptive samples sets')
parser.add_argument('--n_adaptive_sample', nargs='+', type=int, default=10, help='Number of adaptive sampling points per dimension')
parser.add_argument('--balance_weights', nargs='+', type=int, default=1, help='Boolean indicating if loss weights are recalculated at each retraining')
parser.add_argument('--loss_fn', nargs='+', default='MSE', help='Loss functions')
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1,1,1,1,1,1,1,1,1,1], help='Loss weights (more on DIC points)')
parser.add_argument('--num_point_PDE', type=int, default=10000, help='Number of collocation points for PDE evaluation')
parser.add_argument('--num_point_test', type=int, default=10000, help='Number of test points')
parser.add_argument('--sample_notch', type=int, choices=[0, 1], default=0, help='Sample PDE points in the notch region')

parser.add_argument('--net_width', type=int, default=32, help='Width of the network')
parser.add_argument('--net_depth', type=int, default=3, help='Depth of the network')
parser.add_argument('--rank', type=int, default=50, help='Rank for SPINN')
parser.add_argument('--activation', choices=['tanh','sin', 'relu', 'elu'], default='tanh', help='Activation function')
parser.add_argument('--optimizer', choices=['adam'], default='adam', help='Optimizer')
parser.add_argument('--mlp', choices=['mlp', 'modified-mlp'], default='mlp', help='Type of MLP for SPINN')
parser.add_argument('--initialization', choices=['Glorot uniform', 'He normal'], default='Glorot uniform', help='Initialization method')
parser.add_argument('--n_fourier_features', type=int, default=100, help='Number of Fourier features')
parser.add_argument('--SA', type=int, choices=[0, 1], default=0, help='Use self-attention')
parser.add_argument('--SA_update_factor', type=float, default=-1.0, help='Self-attention factor')
parser.add_argument('--SA_init', choices=['constant', 'uniform', 'normal'], default='constant', help='Self-attention initialization')
parser.add_argument('--SA_share_weights', type=int, choices=[0, 1], default=1, help='Self-attention share weights between PDE and material equations')

parser.add_argument('--use_DIC', type=int, choices=[0, 1], default=1, help='Use DIC measurements')
parser.add_argument('--measurments_type', choices=['displacement','strain'], default='strain', help='Type of measurements')
parser.add_argument('--num_measurments', nargs='+', type=int, default=[100,100], help='Number of measurements (should be a perfect square)')
parser.add_argument('--noise_magnitude', type=float, default=1e-6, help='Gaussian noise magnitude (not for DIC simulated)')
parser.add_argument('--u_0', nargs='+', type=float, default=[0,0], help='Displacement scaling factor for Ux and Uy, default(=0) use measurements norm')
parser.add_argument('--s_0', nargs='+', type=float, default=[1,1,1], help='Stress scaling factor for Sxx, Syy, and Sxy')
parser.add_argument('--params_iter_speed', nargs='+', type=float, default=[1,1], help='Scale iteration step for each parameter')
parser.add_argument('--coord_normalization', type=int, default=1, help='Normalize the input coordinates')
parser.add_argument('--stress_integral', type=bool, default=False, help='Impose stress integral to be equal to the side load')
parser.add_argument('--material_law', choices=['isotropic', 'orthotropic'], default='isotropic', help='Material law')
parser.add_argument('--corner_eps', type=float, default=0, help='Corner epsilon for contact boundary conditions')
parser.add_argument('--displacement_BC', type=int, choices=[0, 1], default=0, help='Use displacement BC at the top and bottom edges')

parser.add_argument('--FEM_dataset', type=str, default='3x3mm.dat', help='Path to FEM data')
parser.add_argument('--DIC_dataset_path', type=str, default='no_dataset', help='If default no_dataset, use FEM model for measurements')
parser.add_argument('--DIC_dataset_number', type=int, default=1, help='Only for DIC simulated measurements')
parser.add_argument('--DIC_region', nargs='+', type=float, default=[0, 100, 0, 100], help='Region for DIC measurements [x_min, x_max, y_min, y_max]')
parser.add_argument('--theta', type=float, default=45.0, help='Rotation angle for orthotropic material in degrees')
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
x_max = 100
y_max = 100
notch_diameter = 50
material_law = args.material_law.lower()

# Material parameters (converted to N/mm^2)
if material_law == "isotropic":
    # isotropic plane‐stress
    E       = 52e3     # N/mm^2
    nu      = 0.3
    def constitutive_stress(eps_xx, eps_yy, eps_xy, mat_params=(E, nu)):
        # plane‐stress modified constants
        E, nu = mat_params
        # lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))
        # mu = E / (2 * (1 + nu))

        σ_xx = E/(1-nu**2)* (eps_xx + nu * eps_yy)
        σ_yy = E/(1-nu**2)* (eps_yy + nu * eps_xx)
        σ_xy = E/(1+nu) * eps_xy   

        return σ_xx, σ_yy, σ_xy

elif material_law == "orthotropic":
    # orthotropic plane‐stress
    Q11, Q22, Q12, Q66 = 41e3, 10.3e3, 3.1e3, 4e3   # N/mm^2
    theta = np.deg2rad(args.theta)  # Convert angle to radians for calculations

    def constitutive_stress(eps_xx, eps_yy, eps_xy):
        # plane‐stress modified constants
        σ_xx = Q11*eps_xx + Q12*eps_yy
        σ_yy = Q12*eps_xx + Q22*eps_yy
        σ_xy = 2*Q66*eps_xy
        return σ_xx, σ_yy, σ_xy
    def constitutive_stress(eps_xx, eps_yy, eps_xy):

        # (N,2,2) strain tensor from 3 components
        eps = jnp.stack([eps_xx, eps_xy, eps_xy, eps_yy], axis=-1).reshape(-1, 2, 2)

        # Rotation matrices (2,2)
        c, s = jnp.cos(theta), jnp.sin(theta)
        R  = jnp.array([[c, -s],
                        [s,  c]])
        RT = R.T

        # Rotate strain: eps_local = RT @ eps @ R   (N,2,2)
        eps_local = jnp.einsum('ij,njk,kl->nil', RT, eps, R)

        # Stress in material coordinates (N,2,2), directly built with reshape
        sigma_local = jnp.stack([
            Q11*eps_local[:,0,0] + Q12*eps_local[:,1,1],   # σ11
            2*Q66*eps_local[:,0,1],                        # σ12
            2*Q66*eps_local[:,0,1],                        # σ21
            Q12*eps_local[:,0,0] + Q22*eps_local[:,1,1]    # σ22
        ], axis=-1).reshape(-1, 2, 2)

        # Rotate back to global
        sigma_global = jnp.einsum('ij,njk,kl->nil', R, sigma_local, RT)

        return sigma_global[:,0,0], sigma_global[:,1,1], sigma_global[:,0,1]

# Load
pstress = 50.0
uy_top = pstress * x_max / E  if material_law == 'isotropic' else pstress* x_max / Q22

sin = dde.backend.sin
cos = dde.backend.cos
stack = dde.backend.stack

# Create trainable scaling factors (one per parameter)
# params_factor = [dde.Variable(1 / s) for s in args.params_iter_speed]
# trainable_variables = params_factor

# =============================================================================
# 4.1 Load geometry mapping
# =============================================================================
nx=60
ny=88
dir_path = os.path.dirname(os.path.realpath(__file__))
Xp = np.loadtxt(os.path.join(dir_path, f"deep_notched_{nx}x{ny}.txt"))

# Interpolate mapping
X_map_points = Xp[:, 0].reshape((ny, nx)).T
Y_map_points = Xp[:, 1].reshape((ny, nx)).T

def coordMap(x, X_map = X_map_points, Y_map = Y_map_points, x_max = x_max, y_max=y_max, padding=1e-6):
    x_pos = x[0] / x_max * (X_map.shape[0]-1) * (1-2*padding) + padding
    y_pos = x[1] / y_max * (Y_map.shape[1]-1) * (1-2*padding) + padding
    xm = jax.scipy.ndimage.map_coordinates(X_map,
           [x_pos, y_pos], order=1, mode='nearest')
    ym = jax.scipy.ndimage.map_coordinates(Y_map,
           [x_pos, y_pos], order=1, mode='nearest')
    return jnp.stack([xm, ym])

def tensMap(tens, x):
    J = jax.jacobian(coordMap)(x)
    J_inv = jnp.linalg.inv(J)
    return tens @ J_inv

def calcNormal(x):
    n = jnp.array([-1, 0])
    n_mapped = tensMap(n, x)
    return n_mapped/jnp.linalg.norm(n_mapped)

# =============================================================================
# 4.2 Load FEM Data and Build Interpolation Functions
# =============================================================================
n_rows = 100
n_cols = 100

FEM_dataset = f"{material_law}_{n_rows}x{n_cols}{f'_{args.theta}deg' if material_law == 'orthotropic' else ''}.dat"
dir_path = os.path.dirname(os.path.realpath(__file__))
fem_file = os.path.join(dir_path, r"data_fem", FEM_dataset)

data = np.loadtxt(fem_file)
x_val      = data[:, 0]
y_val      = data[:, 1]
u_val      = data[:, 2:4]
strain_val = data[:, 4:7]
stress_val = data[:, 7:10]
solution_val = np.hstack((u_val, stress_val))

# Interpolate solution
x_grid = np.linspace(0, x_max, n_cols)
y_grid = np.linspace(0, y_max, n_rows)

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
x_integral = np.linspace(0, x_max, n_integral).reshape(-1, 1)
y_integral = np.linspace(0, y_max, n_integral).reshape(-1, 1)
X_integral = [x_integral, y_integral]

S_yy_fem = solution_fn(X_integral)[:, 3:4].reshape(n_integral, n_integral)
x_integral_mesh = transform_coords(X_integral)
x_integral_mesh = jax.vmap(coordMap)(x_integral_mesh)[:,0].reshape(n_integral, n_integral)
S_yy_integral_fem = np.trapezoid(S_yy_fem, x_integral_mesh, axis=0)
p_top = S_yy_integral_fem.mean()

def integral_stress(inputs, outputs, X):
    x = transform_coords(inputs)
    x_mesh = jax.vmap(coordMap)(x)[:,0].reshape((inputs[0].shape[0], inputs[1].shape[0]))

    Syy = outputs[0][:, 3:4].reshape(x_mesh.shape)
    return jnp.trapezoid(Syy, x_mesh, axis=0)

Integral_BC = dde.PointSetOperatorBC(X_integral, p_top, integral_stress)
bcs = [Integral_BC]

# =============================================================================
# 5.1 Setup Free Surface Boundary Conditions
# =============================================================================
n_free = 400
y_free = jnp.linspace(0, x_max, n_free)
X_free = jnp.stack((jnp.zeros(n_free), y_free), axis=1)
notch_dist = (y_max - notch_diameter) / 2

mask = (notch_dist < jax.vmap(coordMap)(X_free)[:, 1]) & (jax.vmap(coordMap)(X_free)[:, 1] < y_max- notch_dist)
X_free = X_free[mask]

X_free_left = [jnp.array([0]).reshape(-1, 1), X_free[:, 1].reshape(-1, 1)]
X_free_right = [jnp.array([x_max]).reshape(-1, 1), X_free[:, 1].reshape(-1, 1)]

def free_surface_balance(inputs, outputs, X):
    if isinstance(inputs, list):
        inputs = transform_coords(inputs)
    outputs = outputs[0]
    normal = jax.vmap(calcNormal)(inputs)
    normal_x, normal_y = normal[:,0], normal[:,1]
    Sxx = outputs[:, 2]
    Syy = outputs[:, 3]
    Sxy = outputs[:, 4]

    balance_x = Sxx * normal_x + Sxy * normal_y
    balance_y = Sxy * normal_x + Syy * normal_y
    return jnp.abs(balance_x) + jnp.abs(balance_y)

Free_BC_left = dde.PointSetOperatorBC(X_free_left, 0, free_surface_balance)
Free_BC_right = dde.PointSetOperatorBC(X_free_right, 0, free_surface_balance)
bcs += [Free_BC_left, Free_BC_right]

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
        X_DIC_input = [np.linspace(args.DIC_region[0], args.DIC_region[1], args.num_measurments[0]).reshape(-1, 1),
                       np.linspace(args.DIC_region[2], args.DIC_region[3], args.num_measurments[1]).reshape(-1, 1)]
        DIC_data = solution_fn(X_DIC_input)[:, :2]
        DIC_data += np.random.normal(0, args.noise_magnitude, DIC_data.shape)

    DIC_norms = np.mean(np.abs(DIC_data), axis=0) # to normalize the loss
    measure_Ux = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 0:1]/DIC_norms[0],
                                          lambda x, f, x_np: f[0][:, 0:1]/DIC_norms[0])
    measure_Uy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 1:2]/DIC_norms[1],
                                          lambda x, f, x_np: f[0][:, 1:2]/DIC_norms[1])
    if args.use_DIC:
        bcs += [measure_Ux, measure_Uy]

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
        X_DIC_input = [np.linspace(args.DIC_region[0], args.DIC_region[1], args.num_measurments[0]).reshape(-1, 1),
                       np.linspace(args.DIC_region[2], args.DIC_region[3], args.num_measurments[1]).reshape(-1, 1)]
        DIC_data = strain_fn(X_DIC_input)
        DIC_data += np.random.normal(0, args.noise_magnitude, DIC_data.shape)
    DIC_norms = np.mean(np.abs(DIC_data), axis=0) # to normalize the loss
    measure_Exx = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 0:1]/DIC_norms[0],
                                           lambda x, f, x_np: strain_from_output(x, f)[:, 0:1]/DIC_norms[0])
    measure_Eyy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 1:2]/DIC_norms[1],
                                           lambda x, f, x_np: strain_from_output(x, f)[:, 1:2]/DIC_norms[1])
    measure_Exy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 2:3]/DIC_norms[2],
                                           lambda x, f, x_np: strain_from_output(x, f)[:, 2:3]/DIC_norms[2])
    if args.use_DIC:
        bcs += [measure_Exx, measure_Eyy, measure_Exy]
else:
    raise ValueError("Invalid measurement type. Choose 'displacement' or 'strain'.")

# Use measurements norm as the default scaling factor
if args.DIC_dataset_path != "no_dataset":
    disp_norms = np.mean(np.abs(np.hstack([Ux_dic, Uy_dic])), axis=0)
else:
    disp_norms = np.mean(np.abs(solution_fn(X_DIC_input)[:, :2]), axis=0)
args.u_0 = [disp_norms[i] if not args.u_0[i] else args.u_0[i] for i in range(2)]

# =============================================================================
# 5.3 Self-Attention Setup
# =============================================================================

trainable_variables = []
if args.SA:
    if args.SA_init == "constant":
        pde_weight = jnp.ones((args.num_point_PDE, 1)) 
        mat_weight = jnp.ones((args.num_point_PDE, 1))
    elif args.SA_init == "uniform":
        key = jax.random.PRNGKey(0)
        pde_weight = jax.random.uniform(key, (args.num_point_PDE, 1))*10
        mat_weight = jax.random.uniform(key, (args.num_point_PDE, 1))*10
    elif args.SA_init == "normal":
        key = jax.random.PRNGKey(0)
        pde_weight = jax.random.normal(key, (args.num_point_PDE, 1))*10 + 10
        mat_weight = jax.random.normal(key, (args.num_point_PDE, 1))*10 + 10
    else:
        raise ValueError("Invalid self-attention initialization. Choose 'constant', 'uniform', or 'normal'.")
    trainable_pde_weight = dde.Variable(pde_weight, update_factor=args.SA_update_factor)
    trainable_variables.append(trainable_pde_weight)
    if not args.SA_share_weights:
        trainable_mat_weight = dde.Variable(mat_weight, update_factor=args.SA_update_factor)
        trainable_variables.append(trainable_mat_weight)

# =============================================================================
# 6. PINN Implementation: Boundary Conditions and PDE Residual
# =============================================================================
# Define the domain geometry
notch_dist = (y_max - notch_diameter) / 2
n_pde = int(args.num_point_PDE**0.5)
x_pde = np.linspace(0, x_max, n_pde).reshape(-1, 1)
y_pde = np.linspace(0, y_max, n_pde).reshape(-1, 1)
x_notch = np.stack((np.linspace(0, 0.2*x_max, int(n_pde/2)),
                    np.linspace(y_max - 0.2*x_max, y_max, int(n_pde/2))), axis=0).reshape(-1, 1)
y_notch = np.linspace(notch_dist, y_max - notch_dist, n_pde).reshape(-1, 1)

X_all = [[x_pde, y_pde], [x_notch, y_notch]] if args.sample_notch else [x_pde, y_pde]

geom = dde.geometry.ListPointCloud(X_all)


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

contact_eps = args.corner_eps * x_max

segs_Sxx = [
    ((0.0, contact_eps), (0.0, y_max - contact_eps)), # left
    ((x_max, contact_eps), (x_max, y_max - contact_eps)), # right
]

segs_Sxy = [
    ((0.0, contact_eps), (0.0, y_max - contact_eps)), # left
    ((x_max, contact_eps), (x_max, y_max - contact_eps)), # right
]

def HardBC(x, f):
    if isinstance(x, (list,tuple)):
        x = transform_coords(x)
    # if args.coord_normalization:
    #     x *= jnp.array([x_max, y_max])  # Normalize coordinates to [0, 1]
    x_mapped = jax.vmap(coordMap)(x)

    Ux = f[:, 0] * x[:, 1] / y_max * (y_max - x[:, 1]) / y_max * args.u_0[0] if args.displacement_BC else f[:, 0] * args.u_0[0]
    Uy = f[:, 1] * x[:, 1] / y_max * (y_max - x[:, 1]) / y_max * args.u_0[1] + uy_top * (x[:, 1] / y_max) if args.displacement_BC else f[:, 1] * args.u_0[1]

    Sxx = f[:, 2] * bc_factor(x_mapped[:, 0], x_mapped[:, 1], segs_Sxx, "C0+")
    Syy = f[:, 3]
    Sxy = f[:, 4] * bc_factor(x_mapped[:, 0], x_mapped[:, 1], segs_Sxy, "C0+")

    return stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)


X_all_grid = transform_coords(X_all)
J_comp2phys = jax.vmap(jax.jacfwd(coordMap))(X_all_grid)
J_phys2comp = jnp.linalg.inv(J_comp2phys)

def pde(x, f, unknowns=None):
    x1, x2 = x[0].reshape(-1, 1), x[1].reshape(-1, 1)
    v_x1 = jnp.ones_like(x1)
    v_x2 = jnp.ones_like(x2)
    # x = transform_coords(x)

    # J_nn = jax.vmap(jax.jacfwd(f[1]))(x)
    J_nn_x1 = jax.jvp(lambda x1: f[1]((x1, x2)), (x1,), (v_x1,))[1]
    J_nn_x2 = jax.jvp(lambda x2: f[1]((x1, x2)), (x2,), (v_x2,))[1]
    J_nn = jnp.stack([J_nn_x1, J_nn_x2], axis=2)  # (N,5,2)

    # J_comp2phys = jax.vmap(jax.jacfwd(coordMap))(x)
    # J_phys2comp = jnp.linalg.inv(J_comp2phys)

    J = jnp.einsum("ijk,ikl->ijl", J_nn, J_phys2comp)

    E_xx = J[:, 0, 0]
    E_yy = J[:, 1, 1]
    E_xy = 0.5 * (J[:, 0, 1] + J[:, 1, 0])

    S_xx, S_yy, S_xy = constitutive_stress(E_xx, E_yy, E_xy)

    Sxx_x = J[:, 2, 0]
    Syy_y = J[:, 3, 1]
    Sxy_x = J[:, 4, 0]
    Sxy_y = J[:, 4, 1]

    momentum_x = Sxx_x + Sxy_y 
    momentum_y = Sxy_x + Syy_y 

    stress_x = S_xx - f[0][:, 2]
    stress_y = S_yy - f[0][:, 3]
    stress_xy = S_xy - f[0][:, 4]

    pde_losses = [momentum_x, momentum_y]
    mat_losses = [stress_x, stress_y, stress_xy]
    if args.SA:
        pde_w = unknowns[0].flatten()
        mat_w = pde_w if args.SA_share_weights else unknowns[1].flatten()
        pde_losses = [pde_w * loss for loss in pde_losses]
        mat_losses = [mat_w * loss for loss in mat_losses]
    return pde_losses + mat_losses

def input_scaling(x):
    """
    Scale the input coordinates to the range [0, 1].
    """
    if isinstance(x, (list,tuple)):
        return [x[0]/x_max, x[1]/y_max]
    else:
        return x / np.array([x_max, y_max])

# =============================================================================
# 7. Fourier Features
# =============================================================================

def list_handler(func):
    def wrapper(x, *args, **kwargs):
        if isinstance(x, (list, tuple)):
            return [func(xi, *args, **kwargs) for xi in x]
        return func(x, *args, **kwargs)
    return wrapper

@list_handler
def fourier_features_transform(x, sigma=10, num_features=args.n_fourier_features):
    """Generate Fourier features for input x."""
    kernel = jax.random.normal(
        jax.random.PRNGKey(0), (x.shape[-1], num_features)
    ) * sigma
    y = jnp.concatenate(
        [x, jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
    )
    return y

# =============================================================================
# 7. Define Neural Network, Data, and Model
# =============================================================================
layers = [2] + [args.net_width] * args.net_depth + [args.rank] + [5]
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
if args.coord_normalization and (args.n_fourier_features > 0):
    net.apply_feature_transform(lambda x: fourier_features_transform(input_scaling(x)))
elif args.coord_normalization:
    net.apply_feature_transform(input_scaling)
elif args.n_fourier_features > 0:
    net.apply_feature_transform(fourier_features_transform)
# if args.coord_normalization:
#     net.apply_feature_transform(input_scaling)

net.apply_output_transform(HardBC)

model = dde.Model(data, net)
model.compile(args.optimizer, lr=args.lr[0], metrics=["l2 relative error"],
              loss_weights=args.loss_weights, loss=args.loss_fn,
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
n_iter_str = sum(args.n_iter) if isinstance(args.n_iter, list) else args.n_iter
folder_name = f"{dic_prefix}{args.material_law}_{num_measurments_str}_{noise_prefix}_{args.available_time if args.available_time else n_iter_str}{'min' if args.available_time else 'iter'}"
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

if args.SA:
    callbacks.append(dde.callbacks.VariableArray(trainable_variables, period=args.log_every,
                                               filename=os.path.join(new_folder_path, "pde_weight_history.dat"),
                                               precision=6, array_position=0))
    if not args.SA_share_weights:
        callbacks.append(dde.callbacks.VariableArray(trainable_variables, period=args.log_every,
                                                   filename=os.path.join(new_folder_path, "mat_weight_history.dat"),
                                                   precision=6, array_position=1))

# Log the history of the output fields
def output_log(x, output, field):
    if field in ['Ux', 'Uy', 'Sxx', 'Syy', 'Sxy']:
        return output[0][:, ['Ux', 'Uy', 'Sxx', 'Syy', 'Sxy'].index(field)]
    if field in ['Exx', 'Eyy', 'Exy']:
        return strain_from_output(x, output)[:, ['Exx', 'Eyy', 'Exy'].index(field)]
    raise ValueError(f"Invalid field name: {field}")
        
X_plot = [np.linspace(0, x_max, 100).reshape(-1, 1), np.linspace(0, y_max, 100).reshape(-1, 1)]
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
# 9.1 Adaptive sampling
# =============================================================================
domain = np.array([[0, x_max], [0, y_max]])
def adaptive_sampling_grid(domain, n, loss_fun, k=1, c=1, n_grid=200, 
                            random_state=None):
    """
    Choose n x-coordinates and n y-coordinates so that the n×n grid they
    generate (via Cartesian product) lies in the regions of highest loss.
    """
    rng = np.random.default_rng(random_state)

    # Build a random trial grid of shape (n_rand, n_rand)
    x_trial = rng.uniform(domain[0, 0], domain[0, 1], n_grid).reshape(-1, 1)  
    y_trial = rng.uniform(domain[1, 0], domain[1, 1], n_grid).reshape(-1, 1)

    # Evaluate the loss on every grid point
    loss_flat = loss_fun([x_trial, y_trial])
    loss = loss_flat.reshape(n_grid, n_grid)

    # Convert the loss into row / column scores
    weight = (loss ** k) / np.mean(loss ** k) + c    # emphasise large losses
    row_scores = weight.sum(axis=1)                  # shape (n_rand,)
    col_scores = weight.sum(axis=0)                  # shape (n_rand,)

    row_idx = np.argsort(-row_scores)[:n]
    col_idx = np.argsort(-col_scores)[:n]

    x_sample = np.sort(x_trial[row_idx])   # sort for nicer grids / plots
    y_sample = np.sort(y_trial[col_idx])

    return x_sample, y_sample

def PDE_loss(X):
    pde_loss_val = model.predict([X[0], X[1]], operator=pde)
    return np.sum(np.abs(pde_loss_val), axis=0)

# =============================================================================
# 9. Calculate Loss Weights based on the Gradient of the Loss Function
# =============================================================================
from jax.flatten_util import ravel_pytree
bc_anchors = [X_integral, X_free_left, X_free_right]
if args.use_DIC:
    bc_anchors += [X_DIC_input, X_DIC_input]
pde_anchors = [[x_pde, y_pde]]
all_anchors = bc_anchors + pde_anchors

def loss_function(params,comp=0,inputs=all_anchors):
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
# loss_weights_grads, loss_grads = calc_loss_weights(model)
# new_loss_weights = [w * g for w, g in zip(args.loss_weights, loss_weights_grads)]
# model.compile(args.optimizer, lr=args.lr, metrics=["l2 relative error"],
#               loss_weights=new_loss_weights, loss=args.loss_fn)
            #   external_trainable_variables=trainable_variables)

# =============================================================================
# 10. Training
# =============================================================================
start_time = time.time()
# print(f"E(GPa): {E_init * params_factor[0].value * args.params_iter_speed[0]/1e3:.3f}, nu: {nu_init * params_factor[1].value * args.params_iter_speed[1]:.3f}")
# losshistory, train_state = model.train(iterations=args.n_iter, callbacks=callbacks, display_every=args.log_every)
pde_anchors = [[x_pde, y_pde], [x_notch, y_notch]] if args.sample_notch else [[x_pde, y_pde]]
if isinstance(args.n_iter, int):
    n_iter = [args.n_iter] * args.n_retrain
else:
    n_iter = args.n_iter

if len(args.n_adaptive_sample) != args.n_retrain:
    args.n_adaptive_sample = [args.n_adaptive_sample[0]] * args.n_retrain
    # print("Warning: n_adaptive_sample length does not match n_retrain. Using the first value for all retrain steps.")

if len(args.balance_weights) != args.n_retrain:
    args.balance_weights = [args.balance_weights[0]] * args.n_retrain
    # print(
    #     "Warning: n_adaptive_sample length does not match n_retrain. "
    #     "Using the first value for all retrain steps."
    # )
if len(args.lr) != args.n_retrain:
    args.lr = [args.lr[0]] * args.n_retrain
    # print("Warning: lr length does not match n_retrain. Using the first value for all retrain steps.")

for i in range(args.n_retrain):
    if i > 0:
        if args.n_adaptive_sample[i] > 0:
            x_sample, y_sample = adaptive_sampling_grid(domain,
                                                        n=args.n_adaptive_sample[i],
                                                        loss_fun=PDE_loss)
            pde_anchors += [[x_sample, y_sample]]
            if len(pde_anchors) > args.max_adaptive_samples+1:
                del pde_anchors[1]
            data.replace_with_anchors(pde_anchors)
            print(f"Adding {x_sample.shape[0]**2} points")

        # Calculate new loss weights if requested
        if args.balance_weights[i]:
            loss_weights_grads, loss_grads = calc_loss_weights(model)
            new_loss_weights = [w * g for w, g in zip(args.loss_weights, loss_weights_grads)]
        else:
            # Only update if not already set to args.loss_weights
            if 'new_loss_weights' not in locals():
                new_loss_weights = args.loss_weights
        model.compile(args.optimizer, lr=args.lr[i], metrics=["l2 relative error"],
                      loss_weights=new_loss_weights, loss=args.loss_fn,
                      external_trainable_variables=trainable_variables)
    print(f"Retrain step {i+1}/{args.n_retrain}, training for {n_iter[i]} iterations with loss weights: {[round(float(w), 2) for w in model.loss_weights]}")
    losshistory, train_state = model.train(iterations=n_iter[i], callbacks=callbacks, display_every=args.log_every)

elapsed = time.time() - start_time

# =============================================================================
# 11. Logging
# =============================================================================
dde.utils.save_loss_history(losshistory, os.path.join(new_folder_path, "loss_history.dat"))

# params_init = [E_init, nu_init]
# variables_history_path = os.path.join(new_folder_path, "variables_history.dat")

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

residual_history = np.array(losshistory.metrics_test)
residual_min = residual_history.min(axis=0)
residual_end = residual_history[-1, :]
residual_mean_end = residual_history[-10:, :].mean(axis=0)


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
        "n_iter": args.n_iter,
        "elapsed": elapsed,
        "iter_per_sec": train_state.epoch / elapsed,
        "backend": dde.backend.backend_name,
        "residual_min": residual_min.tolist(),
        "residual_end": residual_end.tolist(),
        "residual_mean_end": residual_mean_end.tolist(),
        "num_retrain": args.n_retrain,
        "num_adaptive_sample": args.n_adaptive_sample,
        "max_adaptive_samples": args.max_adaptive_samples,
        "balance_weights": args.balance_weights,
        "num_points_pde": sum([anchor[0].shape[0] * anchor[1].shape[0] for anchor in pde_anchors]),
        # "final_loss_weights": model.loss_weights.tolist(),
        "Self-Attention": args.SA,
        "SA_init": args.SA_init if args.SA else None,
        "SA_update_factor": args.SA_update_factor if args.SA else None,
        "SA_share_weights": args.SA_share_weights if args.SA else None,
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
        "fourier_features": args.n_fourier_features,
        "coord_normalization": args.coord_normalization,
        "displacement_BC": args.displacement_BC,
        "loss_function": args.loss_fn,
    }
    problem_info = {
        "x_max": x_max,
        "y_max": y_max,
        "notch_diameter": notch_diameter,
        "material_law": material_law,
        "E": E if material_law == "isotropic" else [Q11, Q22, Q12, Q66],
        "nu": nu if material_law == "isotropic" else None,
        "uy_top": uy_top,
        "theta": args.theta if material_law == "orthotropic" else None,
        "corner_eps": args.corner_eps,
        "coord_normalization": args.coord_normalization,
        # "E_actual": E_actual,
        # "nu_actual": nu_actual,
        # "E_init": E_init,
        # "nu_init": nu_init,
        # "E_final": E_final,
        # "nu_final": nu_final,
    }
    data_info = {
        "use_DIC": args.use_DIC,
        "num_measurments": args.num_measurments,
        "noise_magnitude": args.noise_magnitude,
        "measurments_type": args.measurments_type,
        "DIC_dataset_path": args.DIC_dataset_path,
        "DIC_dataset_number": args.DIC_dataset_number,
        "DIC_region": args.DIC_region,
        "FEM_dataset": FEM_dataset, 
    }
    info = {"system": system_info, "gpu": gpu_info, "execution": execution_info,
            "network": network_info, "problem": problem_info, "data": data_info}
    
    def safe_convert(obj, path="root"):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                out[k] = safe_convert(v, f"{path}.{k}")
            return out
        elif isinstance(obj, (list, tuple)):
            out = []
            for i, v in enumerate(obj):
                try:
                    out.append(safe_convert(v, f"{path}[{i}]"))
                except Exception:
                    warnings.warn(f"Skipping non-serializable entry at {path}[{i}]")
            return out
        else:
            try:
                json.dumps(obj)  # test if serializable
                return obj
            except (TypeError, OverflowError):
                warnings.warn(f"Skipping non-serializable entry at {path} ({type(obj)})")
                return None  # or skip entirely
    info = safe_convert(info)
    with open(fname, "w") as f:
        json.dump(info, f, indent=4)

log_config(os.path.join(new_folder_path, "config.json"))