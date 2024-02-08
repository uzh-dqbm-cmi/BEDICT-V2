import subprocess

def run_inference_script(script_path):
    try:
        print(f"Running script: {script_path}")
        subprocess.run(["python", script_path], check=True)
        print(f"Script {script_path} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")

# Specify the paths to your inference scripts
inference_scripts = [
    "./inference_absolute_efficiency.py",
    "./inference_proportion.py",
    "./overall_outcome_inference.py"
]

# Run scripts sequentially
for script in inference_scripts:
    run_inference_script(script)
