import os
import sys
import subprocess

executable_name = "../pinn_inverse.py"
executable_path = os.path.join(os.path.dirname(__file__), executable_name)
num_runs = 1
material_law = ["isotropic", "orthotropic"][0]  # Change to orthotropic if needed
params_iter_speed = [0.05, 0.1] if material_law == "isotropic" else [0.05]*4
n_retrain = 1
args = {
    # "DIC_dataset_path": r"2_noise_study/data_dic/1noise",
    # "DIC_dataset_number": 1,
    "n_iter": [1000] + [100000]*(n_retrain-1),
    "n_retrain": n_retrain,
    "update_params": [1]*n_retrain,
    "n_adaptive_sample": 10,
    "available_time": 0,
    "num_measurments": [100,100],
    "measurments_type": "displacement",
    "material_law":material_law,  
    "noise_magnitude": 0,
    "loss_weights": [1,1,1,1,1,1,1,1,1e2,1e2],
    "results_path": r"1_inverse/results",
    "log_every": 1000,
    "params_iter_speed": params_iter_speed,
    # "stress_integral": True,
    "num_point_PDE": 100**2,
    # "log_output_fields": [''],
    "corner_eps": 0,  # Angle epsilon for contact boundary conditions
    # "DIC_region": [5, 95, 15, 85],  # Region for DIC measurements [x_min, x_max, y_min, y_max]
    "theta": 45.0,  # Rotation angle for orthotropic material in degrees
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