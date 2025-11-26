import os
import sys
import subprocess

executable_name = "../pinn_forward_sa.py"
executable_path = os.path.join(os.path.dirname(__file__), executable_name)
material_law = ["isotropic", "orthotropic"][1]  # Change to orthotropic if needed
num_runs = 1
n_retrain = 2  # Number of retrainings

args = {
    # "DIC_dataset_path": r"2_noise_study/data_dic/1noise",
    # "DIC_dataset_number": 1,
    "mlp": "mlp",
    "coord_normalization": 1,
    "n_fourier_features": 128,
    "net_width": 32,
    "rank": 32,
    "activation": "sin",
    # "sample_notch": 1,  # Whether to sample PDE points in the notch region
    "n_retrain": n_retrain,
    "n_iter": [1000] + [80000]*(n_retrain-1),
    "lr": [1e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    "max_adaptive_samples": 4,
    "n_adaptive_sample": 0,
    "balance_weights": [1,1] + [1]*(n_retrain-2),
    "available_time": 0,
    "DIC_region": [5, 95, 20, 80],  # [x_min, x_max, y_min, y_max]
    "num_measurments": [100,100],
    "use_DIC": 1,
    "measurments_type": "displacement",
    "material_law": material_law,
    "noise_magnitude": 1e-5,
    "loss_weights": [1,1,1,1,1,1,1,1,1e1,1e1],
    "results_path": r"0_forward/results",
    "log_every": 500,
    "num_point_PDE": 150**2,
    # "log_output_fields": [''],
    "corner_eps": 0.02,  # epsilon for contact boundary conditions
    "theta": 45.0,  # 45.0 ; 0.0
    "displacement_BC": 1,  # 1 to use displacement BC at top and bottom edges
    "SA": 1,
    "SA_update_factor": -1,
    "SA_init": "constant",
    "SA_share_weights": 0,  # Whether to share weights between PDE and material equations
    }

# Flatten the args dictionary into a list of command line arguments
args_list = []
for key, value in args.items():
    if isinstance(value, list):
        args_list.extend([f"--{key}"] + [str(v) for v in value])
    else:
        args_list.append(f"--{key}={value}")

# Run the executable multiple times
print(f"Running executable {executable_path} with arguments: {args_list}")
for run in range(num_runs):
    try:
        print(f"Run number {run+1}/{num_runs}")
        subprocess.check_call([sys.executable, executable_path] + args_list)
    except subprocess.CalledProcessError as e:
        print(f"Run number {run+1}/{num_runs} failed")
        print(e)
        sys.exit(1)