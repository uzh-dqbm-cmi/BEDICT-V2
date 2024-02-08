
# BEDICT-V2:Predicting base editing outcomes with an attention-based deep learning algorithm

<p align="center">
  <img src="./web_application/static/logo.png" alt="Logo" width="400"/>
</p>

---

## Overview

BEDICT-V2 is a deep learning model designed to predict base editing outcomes using an attention-based algorithm. This repository provides the source code and instructions for using the model.
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
├── lib
│   ├── button.d.ts
│   ├── button.js
│   ├── button.js.map
│   ├── button.stories.d.ts
│   ├── button.stories.js
│   ├── button.stories.js.map
│   ├── index.d.ts
│   ├── index.js
│   └── index.js.map
├── package.json
├── src
│   ├── button.stories.tsx
│   ├── button.tsx
│   └── index.ts
└── tsconfig.json
```

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

```

In the above example, the first three backticks start the code block, and the closing three backticks indicate the end of the code block. The text after the closing backticks is back to normal text.
## Usage

### Try the Model on Your Own Sequence

To use the pre-trained model on your own DNA sequence, follow these steps:

1. **Install Dependencies:**
   Make sure you have the required dependencies installed. If you haven't done so, refer to the [Environment Setup](#environme:wq!nt-setup) section.

2. **Load the Model:**
   Download the pretrained model and place it in the right folder

   ```python
   from bedict_v2 import BedictModel

   model = BedictModel()

3. do sth