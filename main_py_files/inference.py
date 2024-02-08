import subprocess

def run_inference_script(script_path, config_file):
    try:
        command = ["python", script_path, config_file]  # Remove '-config_file'
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
        print(f"Script {script_path} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")

# Specify the paths to your inference scripts
inference_scripts = [
    {"script": "inference_absolute_efficiency.py", "config": "config_absolute_efficiency.ini"},
    # Add more scripts if needed
]

# Run scripts sequentially
for script_info in inference_scripts:
    run_inference_script(script_info["script"], script_info["config"])
