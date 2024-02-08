
# BEDICT-V2:Predicting base editing outcomes with an attention-based deep learning algorithm

<p align="center">
  <img src="./web_application/static/logo.png" alt="Logo" width="400"/>
</p>

---

## Overview

BEDICT-V2 is a deep learning model designed to predict base editing outcomes using an attention-based algorithm. This repository provides the source code and instructions for using the model.

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

## Environment Setup

### Step 1: Create a Virtual Environment

Create a virtual environment and install the required dependencies using [Conda](https://docs.conda.io/en/latest/):

```bash
# Create a virtual environment
conda create --name bedict_v2

# Activate the virtual environment
conda activate bedict_v2

# Install dependencies
pip install -r requirements.txt

