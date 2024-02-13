#### inference both for absolute efficiency model and proportion model, outputs the final outcome for each reference sequence including both wild type and non-wild type
import subprocess

def run_inference_script(script, config_file):
    try:
        command = ["python", script, config_file]  
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
        print(f"Script {script} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")

# Specify the paths to your inference scripts
inference_scripts = [
    {"script": "./inference_absolute_efficiency.py", "config": "config_file.ini"},
    #{"script": "./inference_proportion.py", "config": "config_file.ini"},
    {"script": "./overall_outcome_inference.py", "config": "config_file.ini"},
]

# Run scripts sequentially
for script_info in inference_scripts:
    run_inference_script(script_info["script"], script_info["config"])
