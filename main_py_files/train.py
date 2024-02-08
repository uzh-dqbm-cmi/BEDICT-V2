### please specify which editor and which screening method you want to train, otherwise the script will train all the options
import subprocess

# First script
script1 = "./Absolute_efficiency_main.py"
subprocess.run(["python", script1])

# Second script
script2 = "./trainval_test_proportions_main.py"
subprocess.run(["python", script2])