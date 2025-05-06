import os
import sys
import subprocess

executable_path = os.path.join(os.path.dirname(__file__), "../pinn_inverse.py")
num_runs = 1
material_law = 'orthotropic'
nrows, ncols = 100, 120

args = {
    "FEM_dataset": f"{material_law}_{nrows}x{ncols}.dat",
    "coord_normalization": True,
    # "DIC_dataset_path": r"2_noise_study/data_dic/1noise",
    # "DIC_dataset_number": 1,
    "n_iter": 40000,
    "available_time": 2,
    "num_measurments": [12,10],
    "measurments_type": "displacement",
    "noise_magnitude": 0,
    "loss_weights": [1,1,1,1,1,1e2,1e2],
    "results_path": r"0_test/results",
    "log_every": 500,
    "params_iter_speed": [1,1],
    "stress_integral": True,
    "num_point_PDE": 100**2,
    "material_law": material_law,
    # "log_output_fields": [''],
    "coord_normalization": False,
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