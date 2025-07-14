"""
PINN-MT2.0: Deep Notched Plate Implementation

This script implements a Physics-Informed Neural Network (PINN) for the deep notched 
plate problem with coordinate mapping and adaptive sampling.
"""

import os
import time
import json
import argparse
import platform
import subprocess
from functools import wraps
from typing import List, Tuple, Literal

import numpy as np
import jax
import jax.numpy as jnp
import deepxde as dde
from scipy.interpolate import RegularGridInterpolator
from jax.flatten_util import ravel_pytree
import pandas as pd

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
parser = argparse.ArgumentParser(description="Physics Informed Neural Networks for Deep Notched Plate")
parser.add_argument('--n_iter', type=int, default=1000, help='Number of iterations')
parser.add_argument('--log_every', type=int, default=100, help='Log every n steps')
parser.add_argument('--available_time', type=int, default=0, help='Available time in minutes')
parser.add_argument('--log_output_fields', nargs='*', default=['Ux', 'Uy', 'Sxx', 'Syy', 'Sxy'], help='Fields to log')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1,1,1,1,1,1,1,1,1,1], help='Loss weights')
parser.add_argument('--num_point_PDE', type=int, default=200, help='Number of collocation points for PDE evaluation')

parser.add_argument('--net_type', choices=['spinn'], default='spinn', help='Type of network')
parser.add_argument('--bc_type', choices=['hard'], default='hard', help='Type of boundary conditions')
parser.add_argument('--net_width', type=int, default=32, help='Width of the network')
parser.add_argument('--net_depth', type=int, default=5, help='Depth of the network')
parser.add_argument('--activation', choices=['tanh', 'relu', 'elu'], default='tanh', help='Activation function')
parser.add_argument('--optimizer', choices=['adam'], default='adam', help='Optimizer')
parser.add_argument('--mlp', choices=['mlp', 'modified_mlp'], default='mlp', help='Type of MLP for SPINN')
parser.add_argument('--initialization', choices=['Glorot uniform', 'He normal'], default='Glorot uniform', help='Initialization method')

parser.add_argument('--n_DIC', type=int, default=200, help='Number of DIC measurement points')
parser.add_argument('--noise_magnitude', type=float, default=0, help='Gaussian noise magnitude')
parser.add_argument('--u_0', type=float, default=1e-6, help='Displacement scaling factor')
parser.add_argument('--stress_bc', type=bool, default=False, help='Use stress boundary conditions')
parser.add_argument('--material_law', choices=['isotropic', 'orthotropic'], default='isotropic', help='Material law')

parser.add_argument('--save_model', type=bool, default=True, help='Save the trained model')
parser.add_argument('--results_path', type=str, default='results_deep_notched', help='Path to save results')

args = parser.parse_args()

dde.config.set_default_autodiff("forward")

# =============================================================================
# 4. Global Constants, Geometry, and Material Parameters
# =============================================================================
x_max = 1.0
y_max = 1.0
notch_diameter = 0.5
material_law = args.material_law

# Material parameters (converted to N/mm^2)
if material_law == "isotropic":
    # isotropic plane‐stress
    E       = 52e3     # N/mm^2
    nu      = 0.3

    def constitutive_stress(eps_xx, eps_yy, eps_xy):
        # plane‐stress modified constants
        C1 = E/(1 - nu**2)      # factor for normal
        C2 = E/(1 + nu)         # factor for shear
        σ_xx = C1*(eps_xx + nu*eps_yy)
        σ_yy = C1*(eps_yy + nu*eps_xx)
        σ_xy = C2*eps_xy
        return σ_xx, σ_yy, σ_xy

elif material_law == "orthotropic":
    # orthotropic plane‐stress
    Q11, Q22, Q12, Q66 = 41e3, 10.3e3, 3.1e3, 4e3   # N/mm^2

    def constitutive_stress(eps_xx, eps_yy, eps_xy):
        # plane‐stress modified constants
        σ_xx = Q11*eps_xx + Q12*eps_yy
        σ_yy = Q12*eps_xx + Q22*eps_yy
        σ_xy = 2*Q66*eps_xy
        return σ_xx, σ_yy, σ_xy

# Load
pstress = 1.0
uy_top = pstress * x_max / E  if material_law == 'isotropic' else pstress* x_max / Q22

sin = dde.backend.sin
cos = dde.backend.cos
stack = dde.backend.stack

# Load geometry mapping
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

def F(x, phys_xy, X_map, Y_map, x_max, y_max, padding):
    return coordMap(x, X_map, Y_map, x_max, y_max, padding) - phys_xy

def tensMap(tens, x):
    J = jax.jacobian(coordMap)(x)
    J_inv = jnp.linalg.inv(J)
    return tens @ J_inv

def calcNormal(x):
    n = jnp.array([-1, 0])
    n_mapped = tensMap(n, x)
    return n_mapped/jnp.linalg.norm(n_mapped)

# =============================================================================
# 5. Load Solution and Create Interpolation Functions
# =============================================================================
# Load solution
n_mesh_x = 100
n_mesh_y = 100

solution_name = f"{material_law}_{n_mesh_x}x{n_mesh_y}{'_S_bc' if args.stress_bc else ''}.dat"
data = np.loadtxt(os.path.join(dir_path, f"data_fem/{solution_name}"))
X_val = data[:, :2]
u_val = data[:, 2:4]
stress_val = data[:, 7:10]

solution_val = np.hstack((u_val, stress_val))

# Interpolate solution
x_grid = np.linspace(0, x_max, n_mesh_x)
y_grid = np.linspace(0, y_max, n_mesh_y)

interpolators = []
for i in range(solution_val.shape[1]):
    interp = RegularGridInterpolator((x_grid, y_grid), solution_val[:, i].reshape(n_mesh_y, n_mesh_x).T)
    interpolators.append(interp)

def solution_fn(x):
    if args.net_type == "spinn" and isinstance(x, list):
        x_mesh = [x_.ravel() for x_ in jnp.meshgrid(
            jnp.atleast_1d(x[0].squeeze()), 
            jnp.atleast_1d(x[1].squeeze()), 
            indexing="ij"
        )]
        x = stack(x_mesh, axis=-1)
    return np.array([interp((x[:,0], x[:,1])) for interp in interpolators]).T

# =============================================================================
# 6. Geometry Definition and Boundary Conditions
# =============================================================================

# Geometry for deep notched plate
notch_dist = (y_max - notch_diameter) / 2
n_pde = 100
x_all = np.linspace(0, x_max, n_pde).reshape(-1, 1)
y_all = np.linspace(0, y_max, n_pde).reshape(-1, 1)
x_notch = np.stack((np.linspace(0, 0.2, int(n_pde/2)),
                    np.linspace(y_max - 0.2, y_max, int(n_pde/2))), axis=0).reshape(-1, 1)
y_notch = np.linspace(notch_dist, y_max - notch_dist, n_pde).reshape(-1, 1)

geom = dde.geometry.ListPointCloud([[x_all, y_all], 
                                    [x_notch, y_notch]])

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

contact_eps = 0.1 * x_max

segs_Sxx = [
    ((0.0, contact_eps), (0.0, y_max - contact_eps)), # left
    ((x_max, contact_eps), (x_max, y_max - contact_eps)), # right
]

segs_Sxy = [
    ((0.0, contact_eps), (0.0, y_max - contact_eps)), # left
    ((x_max, contact_eps), (x_max, y_max - contact_eps)), # right
]

def HardBC(x, f):
    if args.net_type == "spinn" and isinstance(x, list):
        x_mesh = [x_.ravel() for x_ in jnp.meshgrid(
            jnp.atleast_1d(x[0].squeeze()), 
            jnp.atleast_1d(x[1].squeeze()), 
            indexing="ij"
        )]
        x = stack(x_mesh, axis=-1)
    x_mapped = jax.vmap(coordMap)(x)

    Ux = f[:, 0] * x[:, 1] / y_max * (y_max - x[:, 1]) / y_max * args.u_0
    Uy = f[:, 1] * x[:, 1] / y_max * (y_max - x[:, 1]) / y_max * args.u_0 + uy_top * (x[:, 1] / y_max)

    Sxx = f[:, 2] * bc_factor(x_mapped[:, 0], x_mapped[:, 1], segs_Sxx, "C0+")
    Syy = f[:, 3]
    Sxy = f[:, 4] * bc_factor(x_mapped[:, 0], x_mapped[:, 1], segs_Sxy, "C0+")

    return stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)

def pde(x, f):
    x = transform_coords(x)

    J_nn = jax.vmap(jax.jacfwd(f[1]))(x)
    J_comp2phys = jax.vmap(jax.jacfwd(coordMap))(x)
    J_phys2comp = jnp.linalg.inv(J_comp2phys)

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

    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]

# Integral stress BC
n_integral = 100
x_integral = np.linspace(0, x_max, n_integral).reshape(-1, 1)
y_integral = np.linspace(0, y_max, n_integral).reshape(-1, 1)
X_integral = [x_integral, y_integral]

p_top = 0.7416 

def integral_stress(inputs, outputs, X):
    if args.net_type == "spinn" and isinstance(inputs, list):
        x_mesh = [x_.ravel() for x_ in jnp.meshgrid(
            jnp.atleast_1d(inputs[0].squeeze()), 
            jnp.atleast_1d(inputs[1].squeeze()), 
            indexing="ij"
        )]
        x = stack(x_mesh, axis=-1)
    x_mesh = jax.vmap(coordMap)(x)[:,0].reshape((inputs[0].shape[0], inputs[0].shape[0]))

    Syy = outputs[0][:, 3:4].reshape(x_mesh.shape)
    return jnp.trapezoid(Syy, x_mesh, axis=0)

Integral_BC = dde.PointSetOperatorBC(X_integral, p_top*x_max, integral_stress)

# Free surface BC
n_free = 400
y_free = jnp.linspace(0, x_max, n_free)
X_free = jnp.stack((jnp.zeros(n_free), y_free), axis=1)

mask = (notch_dist < jax.vmap(coordMap)(X_free)[:, 1]) & (jax.vmap(coordMap)(X_free)[:, 1] < y_max- notch_dist)
X_free = X_free[mask]

X_free_left = [jnp.array([0]).reshape(-1, 1), X_free[:, 1].reshape(-1, 1)]
X_free_right = [jnp.array([x_max]).reshape(-1, 1), X_free[:, 1].reshape(-1, 1)]

def free_surface_balance(inputs, outputs, X):
    if args.net_type == "spinn" and isinstance(inputs, list):
        x_mesh = [x_.ravel() for x_ in jnp.meshgrid(
            jnp.atleast_1d(inputs[0].squeeze()), 
            jnp.atleast_1d(inputs[1].squeeze()), 
            indexing="ij"
        )]
        inputs = stack(x_mesh, axis=-1)
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

bcs = [Free_BC_left, Free_BC_right, Integral_BC]

num_boundary = 0
X_DIC_input = [np.linspace(0, x_max, args.n_DIC).reshape(-1, 1)] * 2

DIC_data = solution_fn(X_DIC_input)[:, :2]
DIC_data += np.random.normal(0, args.noise_magnitude, DIC_data.shape)

DIC_norms = np.mean(np.abs(DIC_data), axis=0) # to normalize the loss
measure_Ux = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 0:1]/DIC_norms[0],
                                        lambda x, f, x_np: f[0][:, 0:1]/DIC_norms[0])
measure_Uy = dde.PointSetOperatorBC(X_DIC_input, DIC_data[:, 1:2]/DIC_norms[1],
                                        lambda x, f, x_np: f[0][:, 1:2]/DIC_norms[1])
bcs += [measure_Ux, measure_Uy]
# =============================================================================
# 7. Define Neural Network, Data, and Model
# =============================================================================

def get_num_params(net, input_shape=None):
    if dde.backend.backend_name == "pytorch":
        return sum(p.numel() for p in net.parameters())
    elif dde.backend.backend_name == "paddle":
        return sum(p.numpy().size for p in net.parameters())
    elif dde.backend.backend_name == "jax":
        if input_shape is None:
            raise ValueError("input_shape must be provided for jax backend")
        import jax
        import jax.numpy as jnp

        rng = jax.random.PRNGKey(0)
        return sum(
            p.size for p in jax.tree.leaves(net.init(rng, jnp.ones(input_shape)))
        )

if args.net_type == "spinn":
    layers = [2, args.net_width, args.net_width, args.net_width, 5]
    net = dde.nn.SPINN(layers, args.activation, args.initialization, args.mlp)
    num_point = args.num_point_PDE**2
    total_points = num_point**2 + num_boundary**2
    num_params = get_num_params(net, input_shape=layers[0])
    x_plot = np.linspace(0,x_max,100)
    y_plot = np.linspace(0,y_max,100)
    X_plot = [x_plot.reshape(-1, 1), y_plot.reshape(-1, 1)]

num_test = 10000

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
# 8. Training and Adaptive Sampling
# =============================================================================

def adaptive_sampling_grid(domain, n, loss_fun, k=1, c=1, n_rand=200, 
                           use_probabilistic=True, random_state=None):
    """
    Choose n x-coordinates and n y-coordinates so that the n×n grid they
    generate (via Cartesian product) lies in the regions of highest loss.
    """
    rng = np.random.default_rng(random_state)

    # Build a random trial grid of shape (n_rand, n_rand)
    x_trial = rng.uniform(domain[0, 0], domain[0, 1], n_rand).reshape(-1, 1)  
    y_trial = rng.uniform(domain[1, 0], domain[1, 1], n_rand).reshape(-1, 1)

    Xg, Yg = np.meshgrid(x_trial, y_trial, indexing='ij')   # shape (n_rand, n_rand)

    # Evaluate the loss on every grid point
    loss_flat = loss_fun([x_trial, y_trial])
    loss = loss_flat.reshape(n_rand, n_rand)

    # Convert the loss into row / column scores
    weight = (loss ** k) / np.mean(loss ** k) + c    # emphasise large losses
    row_scores = weight.sum(axis=1)                  # shape (n_rand,)
    col_scores = weight.sum(axis=0)                  # shape (n_rand,)

    if use_probabilistic:
        row_p = row_scores / row_scores.sum()
        col_p = col_scores / col_scores.sum()

        row_idx = rng.choice(n_rand, size=n, replace=False, p=row_p)
        col_idx = rng.choice(n_rand, size=n, replace=False, p=col_p)
    else:
        row_idx = np.argsort(-row_scores)[:n]
        col_idx = np.argsort(-col_scores)[:n]

    x_sample = np.sort(x_trial[row_idx])   # sort for nicer grids / plots
    y_sample = np.sort(y_trial[col_idx])

    return x_sample, y_sample

bc_anchors = [X_free_left, X_free_right, X_integral, X_DIC_input, X_DIC_input]
pde_anchors = [[x_all, y_all]]
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

# Training
losshistory, train_state = model.train(
    iterations=100000, display_every=10000
)

# Adaptive sampling and re-training
domain = np.array([[0, x_max], [0, y_max]])

def PDE_loss(X):
    pde_loss_val = model.predict([X[0], X[1]], operator=pde)
    return np.sum(np.abs(pde_loss_val), axis=0)

n_iter = 10000
log_every = 1000
for i in range(5):
    losshistory, train_state = model.train(
        iterations=n_iter, display_every=log_every
    )
    x_sample, y_sample = adaptive_sampling_grid(domain,
                                        n=10,
                                        loss_fun=PDE_loss,
                                        k=2, c=1,
                                        n_rand=200,
                                        use_probabilistic=False)
    pde_anchors += [[x_sample, y_sample]]
    data.replace_with_anchors(pde_anchors)

    loss_weights_grads, loss_grads = calc_loss_weights(model)
    new_loss_weights = [w * g for w, g in zip(args.loss_weights, loss_weights_grads)]
    
    model.compile(args.optimizer, lr=args.lr, metrics=["l2 relative error"],
                loss_weights=new_loss_weights)