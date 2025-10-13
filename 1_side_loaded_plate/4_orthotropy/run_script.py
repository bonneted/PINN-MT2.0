import os
import sys
import subprocess

executable_path = os.path.join(os.path.dirname(__file__), "../pinn_inverse_ortho.py")
num_runs = 1
material_law = "orthotropic"  # Options: "isotropic", "orthotropic"
fem_dataset = "100x100mm_ortho.dat" if material_law == "orthotropic" else "100x100mm.dat"

args = {
    "material_law": material_law,
    "FEM_dataset": fem_dataset,
    "available_time": 20,
    "measurments_type": "displacement",
    "noise_magnitude": 1e-6,
    "loss_weights": [1,1,1,1,1,1e2,1e2],
    "results_path": "4_orthotropy/results",
    "log_every": 500,
    "num_measurments": [25, 25],
    # "stress_integral": True,
    # "stress_BC": True,
    "num_point_PDE": 200**2,
    # "log_output_fields": [''],
}

# Flatten the args dictionary into a list of command line arguments
args_list = []
for key, value in args.items():
    if isinstance(value, list):
        args_list.extend([f"--{key}"] + [str(v) for v in value])
    else:
        args_list.append(f"--{key}={value}")

# Run the executable multiple times
for run in range(num_runs):
    try:
        print(f"Run number {run+1}/{num_runs}")
        subprocess.check_call([sys.executable, executable_path] + args_list)
    except subprocess.CalledProcessError as e:
        print(f"Run number {run+1}/{num_runs} failed")
        print(e)
        sys.exit(1)