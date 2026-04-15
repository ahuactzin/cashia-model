# Cashia Model

**cashia-model** contains the **machine learning models and model
management utilities** used in the Cashia ecosystem.

This package is responsible for:

-   loading trained credit models
-   managing model containers
-   providing prediction interfaces
-   organizing model resources

It is used primarily by:

-   **CCE (Cashia Credit Engine)** -- for internal credit engine
    computations
-   **cashia-api** -- to expose predictions through REST endpoints

------------------------------------------------------------------------

# 1. Requirements

Before installing `cashia-model`, ensure the following are available:

-   Python **3.11**
-   pip
-   Access to the repositories:

```{=html}
<!-- -->
```
    cashia-core
    cashia-model

Recommended tools:

-   **Anaconda / Miniconda**
-   **Visual Studio Code**

------------------------------------------------------------------------

# 2. Environment Setup

## Option A --- Using Conda (recommended)

Create the environment:

``` bash
conda create -n cashia_env python=3.11
```

Activate it:

``` bash
conda activate cashia_env
```

------------------------------------------------------------------------

## Option B --- Using venv

Create a virtual environment:

``` bash
python -m venv cashia_env
```

Activate it:

### Windows

``` bash
cashia_env\Scripts\activate
```

### Linux / macOS

``` bash
source cashia_env/bin/activate
```

------------------------------------------------------------------------

# 3. Installation

Navigate to the root directory containing the Cashia repositories.

Example:

    cashia
    │
    ├── cashia-core
    ├── cashia-model
    ├── cce
    ├── cashia-api
    └── mlp

Install dependencies in editable mode:

``` bash
pip install -e cashia-core
pip install -e cashia-model
```

Editable mode is recommended during development.

------------------------------------------------------------------------

# 4. Project Structure

Typical structure of the project:

    cashia-model
    │   pyproject.toml
    │   README.md
    │
    └───src
        └───cashia_model
            │   pm_model_rebuilder.py
            │   model_container.py
            │   prediction_manager.py
            │
            ├── models
            ├── utils
            └── resources

### Key modules

  --------------------------------------------------------------------------
  Module                    Description
  ------------------------- ------------------------------------------------
  `pm_model_rebuilder.py`   Reconstructs model objects from serialized files

  `model_container.py`      Container class for managing multiple models

  `prediction_manager.py`   Handles prediction workflows

  `models/`                 Serialized ML models

  `resources/`              configuration resources related to models
  --------------------------------------------------------------------------

------------------------------------------------------------------------

# 5. Model Files

Models are typically stored as **serialized files** such as:

    .sav
    .pkl

Examples:

    Random_Forest_NV_Agt.sav
    Random_Forest_NV_CC.sav
    Random_Forest_RNV_Agt.sav
    Random_Forest_RNV_CC.sav

These models are loaded dynamically using utilities in the package.

------------------------------------------------------------------------

# 6. Model Container

A central component of this package is the **ModelsContainer**, which
loads and manages all available models.

Example usage:

``` python
from cashia_model.model_container import ModelsContainer

models = ModelsContainer()

prediction = models.predict(
    model_name="NV_Agt",
    input_data=data
)
```

The container ensures that models are loaded once and reused
efficiently.

------------------------------------------------------------------------

# 7. Loading Serialized Models

Serialized models are typically loaded using Python's `pickle` module.

Example:

``` python
import pickle

with open("Random_Forest_NV_Agt.sav", "rb") as f:
    model = pickle.load(f)
```

In practice, `cashia-model` wraps this process inside helper utilities
to standardize loading.

------------------------------------------------------------------------

# 8. Interaction with cashia-core

Model resources are accessed using utilities provided by
**cashia-core**.

Example:

``` python
from cashia_core.common_tools.storage import get_storage
from cashia_core.common_tools.configuration.resource_keys import get_model_path

storage = get_storage()

model_path = get_model_path("nv_agt")
model_bytes = storage.read_bytes(model_path)
```

This allows models to be loaded from:

-   local storage
-   cloud storage (e.g., S3)

without modifying model code.

------------------------------------------------------------------------

# 9. Example Prediction

Example simplified prediction flow:

``` python
from cashia_model.model_container import ModelsContainer

models = ModelsContainer()

result = models.predict(
    model_name="NV_CC",
    input_data=application_features
)

print(result)
```

This interface is typically used by:

-   the **CCE engine**
-   the **Cashia API**

------------------------------------------------------------------------

# 10. Development Notes

## Editable install

During development install with:

``` bash
pip install -e cashia-model
```

This allows modifications to the model package without reinstalling.

------------------------------------------------------------------------

## Reinstall after dependency changes

If dependencies change in `pyproject.toml`:

``` bash
pip install -e cashia-model --upgrade
```

------------------------------------------------------------------------

# 11. Troubleshooting

### Pickle compatibility errors

Serialized models depend on specific library versions.

Typical compatible versions may include:

    pandas==1.5.3
    numpy==1.26.0
    scikit-learn==1.1.3

If loading fails, verify that the correct versions are installed.

------------------------------------------------------------------------

### Missing model files

If a model cannot be loaded, verify that:

-   the model file exists
-   the resource path is correctly configured
-   storage backend settings are correct

------------------------------------------------------------------------

# 12. Role in the Cashia Architecture

Within the Cashia architecture:

  Component      Responsibility
  -------------- ---------------------------
  cashia-core    shared utilities
  cashia-model   machine learning models
  cce            credit computation engine
  cashia-api     REST API interface

`cashia-model` acts as the **model layer** between the core
infrastructure and the credit engine.

------------------------------------------------------------------------

# 13. Future Improvements

Possible future enhancements include:

-   automatic model reloading

------------------------------------------------------------------------

# Author

Juan Manuel Ahuactzin\
Cashia Project
