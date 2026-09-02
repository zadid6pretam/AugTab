# AugTab: Learnable Feature Augmentation for Low-Dimensional Tabular Data (ECML PKDD 2026 Research Track)

![Status](https://img.shields.io/badge/Status-Accepted-success)
![ECML PKDD 2026](https://img.shields.io/badge/ECML%20PKDD-2026-purple)
![Track](https://img.shields.io/badge/Track-Research%20Track-blue)
[![Paper](https://img.shields.io/badge/Paper-In%20Press-red)](https://doi.org/10.1007/978-3-032-37670-1_24)
![Feature Augmentation](https://img.shields.io/badge/Focus-Feature%20Augmentation-blueviolet)
![Low-Dimensional Tabular Data](https://img.shields.io/badge/Data-Low--Dimensional%20Tabular%20Data-teal)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-brightgreen)

<p align="center">
  <img src="AugTab_Architecture.png" alt="AugTab Architecture" width="1000">
</p>

AugTab is a learnable feature augmentation framework for **low-dimensional tabular learning**. It introduces a **Feature Augmentation Layer (FAL)** that enriches the original feature space through complementary **nonlinear projection, explicit cross-interaction, and gated recombination** branches, allowing feature augmentation to be learned jointly with the downstream predictor rather than treated as fixed preprocessing or manual feature engineering. FAL combines a nonlinear MLP branch with a cross-interaction branch to construct candidate augmented features, followed by a learned projection and **input-adaptive gating** mechanism that selectively activates useful augmented dimensions for each sample. The resulting augmented representation retains the original features while appending task-aligned learned features, enabling the model to increase expressive capacity without discarding the original input information.

To keep this expansion controlled, AugTab incorporates regularization for **augmentation sparsity, feature diversity, orthogonality, computational budget, and robustness to distribution shift**. These objectives discourage redundant augmented features, encourage complementarity with the original feature space, and regulate the expected number of active augmented dimensions. The framework also supports perturbation-based drift consistency and missingness-aware learning. Although the Feature Augmentation Layer is designed to be **backbone-agnostic**, the ECML PKDD 2026 implementation instantiates AugTab with a lightweight **MLP backbone**. Theoretical analysis further connects learnable augmentation with increased expressivity, improved input conditioning, and generalization under a bounded gating budget. Across **15 low-dimensional classification and regression datasets**, AugTab achieves the best average rank across classification tasks and the top result on all evaluated regression tasks, with comparisons spanning **54 classification baselines and 16 regression baselines**.

## Overview

**AugTab** is a tabular learning framework for problems where the original feature space is relatively low-dimensional and may provide limited expressive structure because of noise, heterogeneity, weak feature interactions, missingness, or distribution shift.

The key idea is to make **feature augmentation itself learnable**. Instead of applying a fixed polynomial expansion, manually constructing feature interactions, or relying solely on the downstream model to discover useful representations, AugTab inserts a trainable **Feature Augmentation Layer (FAL)** before the prediction backbone.

FAL constructs augmented features using:

- **Cross-interaction branch** - learns explicit second-order feature interactions through factorized transformations.
- **Nonlinear MLP branch** - learns nonlinear feature lifts from the original input.
- **Learned projection** - combines and re-mixes complementary augmented representations.
- **Input-adaptive gating** - determines which augmented dimensions should be activated for each sample.
- **Budget-aware regularization** - controls the effective augmentation width and computational cost.
- **Diversity and orthogonality regularization** - encourages augmented features to remain complementary rather than redundant.
- **Drift consistency** - promotes stable augmented representations under distribution shifts and feature perturbations.

For an input feature vector $x$, AugTab preserves the original features and concatenates them with the gated learned augmentation:

$$
\tilde{x} = x \oplus \big(g(x) \odot z(x)\big)
$$

where $z(x)$ represents the candidate augmented features and $g(x)$ provides input-dependent gates controlling their activation.

AugTab is **not tied to a specific low-dimensional dataset**. The Feature Augmentation Layer is designed as a modular front end that can, in principle, be combined with different downstream tabular architectures. In the experiments reported in the paper, AugTab is instantiated with an **MLP backbone**.

## Citation

Al Zadid Sultan Bin Habib, Md Younus Ahamed, Md Asif Bin Syed, Md Samiul Islam, Muntasir Tabasum, Tanpia Tasnim, and Md. Ekramul Islam. **“AugTab: Learnable Feature Augmentation for Low-Dimensional Tabular Data.”** In *Machine Learning and Knowledge Discovery in Databases. Research Track (ECML PKDD 2026)*, Springer, 2027. https://doi.org/10.1007/978-3-032-37670-1_24

BibTeX:

```bibtex
@inproceedings{habib2026augtab,
  title     = {AugTab: Learnable Feature Augmentation for Low-Dimensional Tabular Data},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Syed, Md Asif Bin and Islam, Md Samiul and Tabasum, Muntasir and Tasnim, Tanpia and Islam, Md. Ekramul},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Research Track},
  year      = {2027},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-032-37670-1_24}
}
```

- **Paper:** https://doi.org/10.1007/978-3-032-37670-1_24 *(In Press)*
- **Project Page:** https://www.zadidhabib.com/augtab.html
- **GitHub:** https://github.com/zadid6pretam/AugTab

This repository contains the official implementation of **AugTab** together with example notebooks for reproducing the training workflow, hyperparameter tuning, evaluation, and package installation.

The repository currently includes:

- `augtab/augtab.py` — core implementation of the **AugTab** framework, including the Feature Augmentation Layer (FAL), MLP backbone, regularization components, and task-specific interfaces for binary classification, multiclass classification, and regression.
- `augtab/__init__.py` — package initialization file exposing the main AugTab classes and configuration objects through the `augtab` package.
- `AugTab Try.ipynb` — example notebook demonstrating how to use AugTab with **Optuna-based hyperparameter tuning** and **5-fold cross-validation**, reporting **mean accuracy ± standard deviation** on the **Water Potability** dataset.
- `PIP_Install_Check.ipynb` — lightweight notebook for verifying the **PyPI installation**, package imports, and basic AugTab functionality after installation with `pip install augtab`.
- `pyproject.toml` — modern Python project and build-system configuration used for packaging and PyPI distribution.
- `setup.cfg` — package metadata and configuration for installation and distribution.
- `requirements.txt` — Python dependencies required to run AugTab and the accompanying experiments.

The example notebook is written so that the dataset file and target column can be changed easily, allowing the same workflow to be reused for other low-dimensional tabular datasets.

In addition, `AugTab Try.ipynb` contains supplementary diagnostic and analysis code used to further inspect AugTab's behavior during training and evaluation.

---

## Repository Contents

```text
.
├── augtab/
│   ├── __init__.py
│   └── augtab.py
│
├── AugTab Try.ipynb
├── PIP_Install_Check.ipynb
├── pyproject.toml
├── setup.cfg
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

### Package Structure

The installable Python package is contained in the `augtab/` directory:

```text
augtab/
├── __init__.py
└── augtab.py
```

The public API can therefore be imported directly from the package:

```python
from augtab import (
    AugTabClassifier,
    AugTabMulti,
    AugTabRegressor,
)
```

The three primary interfaces correspond to:

- `AugTabClassifier` — binary classification
- `AugTabMulti` — multiclass classification
- `AugTabRegressor` — regression

Lower-level configuration and model components are also exposed for users who want more control over the AugTab architecture and training configuration.

### Main dependencies

The repository uses the following main dependencies:

```
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
optuna>=3.0
torch>=2.0
matplotlib>=3.6
jupyter>=1.0
notebook>=6.5
ipykernel>=6.0
tqdm>=4.64
scipy>=1.9
```
## Installation

You can install **AugTab** in several ways depending on your workflow.

---

### Option 1: Clone the Repository (Recommended for Development)

```bash
git clone https://github.com/zadid6pretam/AugTab.git
cd AugTab
pip install -r requirements.txt
pip install -e .
```

- This is the recommended option if you want to modify the source code, run the provided experiment notebooks, reproduce the Optuna-based tuning workflow, or develop additional AugTab extensions.
- Editable installation (`-e`) allows changes made inside the local `augtab/` package to be reflected immediately without reinstalling the package.

---

### Option 2: Install Directly from GitHub (No Cloning Needed)

```bash
pip install "git+https://github.com/zadid6pretam/AugTab.git"
```

- This installs the latest version of **AugTab** directly from the GitHub repository.

After installation, the task-specific AugTab interfaces can be imported as:

```python
from augtab import AugTabClassifier, AugTabMulti, AugTabRegressor
```

---

### Option 3: Use a Virtual Environment

```bash
python -m venv augtab-env

# macOS / Linux
source augtab-env/bin/activate

# Windows
# augtab-env\Scripts\activate

git clone https://github.com/zadid6pretam/AugTab.git
cd AugTab
pip install -r requirements.txt
pip install -e .
```

- Using a virtual environment is recommended to keep **AugTab** and its dependencies isolated from other Python projects.
- Once activated, all AugTab dependencies and experiments will run within the isolated environment.

---

### Option 4: Local Install Without Editable Mode

```bash
git clone https://github.com/zadid6pretam/AugTab.git
cd AugTab
pip install -r requirements.txt
pip install .
```

- This performs a standard local installation of AugTab.
- Unlike editable mode, subsequent changes to the source code require reinstalling the package.

After installation:

```python
from augtab import AugTabClassifier, AugTabMulti, AugTabRegressor
```

---

### Option 5: Install from PyPI

```bash
pip install augtab
```

After installation, the main AugTab interfaces can be imported as:

```python
from augtab import AugTabClassifier, AugTabMulti, AugTabRegressor
```

The three high-level interfaces correspond to:

- `AugTabClassifier` — binary classification
- `AugTabMulti` — multiclass classification
- `AugTabRegressor` — regression

For example:

```python
from augtab import AugTabClassifier

model = AugTabClassifier(
    d_features=10,
    k_aug=32,
    device="cuda"
)
```

Lower-level configuration and model components are also directly available:

```python
from augtab import (
    FALConfig,
    BackboneConfig,
    RegularizerConfig,
    AugTabConfig,
    AugTabCore,
)
```

A lightweight installation check is provided in:

```text
PIP_Install_Check.ipynb
```

which can be used to verify the PyPI installation, package imports, and basic AugTab functionality.
