
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


### Run Inference on Custom Sequences

You can use the pre-trained BEDICT-V2 models to run inference on your own DNA sequences. Choose between running locally via a notebook or using our web app.

---

### 🧪 Option 1: Local Inference Using the Notebook

1. **Prepare your input file:**

   - Create an Excel file with:
     - **Target sequences** (20 bases long)
     - **PAM sequences** (4 bases long)
   - Place this file in the `dataset/` directory.

2. **Open the notebook:**

   Navigate to the `notebooks/` folder and open `Inference_user_defined_sequence.ipynb`.

3. **Configure your run:**

   In the notebook, specify:
   - The **input Excel file name**
   - The **editor name** (e.g., ABE8e-NG)
   - Whether you're predicting **in vivo** or **in vitro**

4. **Run inference:**

   The notebook will automatically run:
   - The **absolute efficiency model**
   - The **proportional model**

   It will then merge the predictions into a final result table.



### 🌐 Option 2: Use the Web App (Easiest)

The easiest way to use BEDICT-V2 is through our [web app](https://go.bedict.app/). Just upload your sequences and get results instantly — no setup required!



### 📦 Note on Pre-trained Models

Pre-trained models are already included in the repository under corresponding folders, such as BEDICT-V2/absolute_efficiency_model/output/...


## Train the Model on Your Own Dataset

To deploy BEDICT-V2 on your own dataset (e.g., screening data), follow the steps below:

---

### 1. **Prepare the Data**

An example dataset is provided in the `dataset/` folder. Your dataset should be in Excel format and include the following columns:

- **Target protospacer** (20 bases)
- **PAM sequence** (4 bases)
- **Outcome sequence** (20 bases)

---

### 2. **Pre-process the Data**

Use the preprocessing script to convert your Excel input into model-ready formats:

```bash
python main_py_files/generate_two_stage_model_data.py
```

## License
[License](LICENSE)
