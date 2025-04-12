
# BEDICT-V2:Predicting base editing outcomes with an attention-based deep learning algorithm

<p align="center">
  <img src="./web_application/static/logo.png" alt="Logo" width="400"/>
</p>

---

## Overview

BEDICT-V2 is a deep learning model designed to predict base editing outcomes using an attention-based algorithm. This repository provides the source code and instructions for using the model. We also have a [web app you can try out here](https://go.bedict.app/).
https://go.bedict.app/
<p align="center">
  <img src="./web_application/static/model.png" alt="Logo" width="800"/>
</p>
---

## Table of Contents

- [Environment Setup](#environment-setup)
  - [Step 1: Create a Virtual Environment](#step-1-create-a-virtual-environment)
- [Usage](#usage)
  - [Try the Model on Your Own Sequence](#try-the-model-on-your-own-sequence)
  - [Retrain the Model](#retrain-the-model)
- [Contributing](#contributing)
- [License](#license)

---
## The folder structure:
```
packages/button
├── absolute_efficiency_model
│   ├── models
│   ├── output
│   └── src
├── dataset
├── main_py_files
│   ├── train.py
│   ├── ....
│   └──inference.py
├── dataset
├── notebooks
├── proportion_model
│   ├── output
│   └── src
├── utils
├── web_application
│   ├── templates
│   ├── static
│   └── app.y
├── README.md
└── requirment.txt
```

## Environment Setup

### Set up the environment

Create a virtual environment and install the required dependencies using [Conda](https://docs.conda.io/en/latest/):

```bash
# Create a virtual environment
conda create --name bedict_v2

# Activate the virtual environment
conda activate bedict_v2

# install python
conda install -c anaconda python=3.10

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install dependencies
pip install -r requirements.txt

```

## Usage

## How to Run Inference on Custom Sequences

To use the pre-trained model on your own DNA sequence, follow these steps:

1. **Install Dependencies:**
   Make sure you have the required dependencies installed. If you haven't done so, refer to the [Environment Setup](#environme:wq!nt-setup) section.

2. **Get the trained model**
   Trained model is already placed in the corresponding folders, such as (BEDICT-V2/absolute_efficiency_model/output/CNN_v2/ABE8e-NG/protospacer_PAM/train_val) 


### 🧪 Option 1: Use the Notebook (Local Inference)

1. Place the Excel file containing:
   - **Target sequences** (20 bases long)
   - **PAM sequences** (4 bases long)  
   in the `dataset/` folder.

2. Navigate to the `notebook/` directory and open the notebook named `Inference_user_defined_sequence.ipynb`.

3. Inside the notebook, you can specify:
   - The **data file name**
   - The **editor name**
   - Whether you want to predict **in vivo** or **in vitro**

4. The notebook will generate predictions using both the **absolute efficiency model** and the **proportional model**, and produce the **final merged results** automatically.

---

### 🌐 Option 2: Try It Online (Easiest)

You can also use our [web app](https://go.bedict.app/) for a quick and user-friendly experience — no setup required!


### Train the model on your own dataset

To deploy our model on your dataset, where you will train our model on your screening data, follow these steps:

1. **Prepare the data:**

There is an example data you will find in the dataset store in exel file, where it includes columsn with target protospacer (20 bases), pam information (four bases), and outcome sequence (20 bases)

2.  **Pre-process the data:**

Go to the foler main_py_files and run generate_two_stage_model_data, input the exel file name

```bash
python inference.py
```

2.  **Train the model:**

Go to the foler main_py_files and call inference.py file, before that, you can select the method (in vivo or in vitro), editor in the config file and run inference.py



2.  **Infer the model:**

Go to the foler main_py_files and call inference.py file, before that, you can select the method (in vivo or in vitro), editor in the config file and run inference.py

## License
[License](LICENSE)
