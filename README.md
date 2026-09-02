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

- `augtab/augtab.py` - core implementation of the **AugTab** framework, including the Feature Augmentation Layer (FAL), MLP backbone, regularization components, and task-specific interfaces for binary classification, multiclass classification, and regression.
- `augtab/__init__.py` - package initialization file exposing the main AugTab classes and configuration objects through the `augtab` package.
- `AugTab Try.ipynb` - example notebook demonstrating how to use AugTab with **Optuna-based hyperparameter tuning** and **5-fold cross-validation**, reporting **mean accuracy ± standard deviation** on the **Water Potability** dataset.
- `PIP_Install_Check.ipynb` - lightweight notebook for verifying the **PyPI installation**, package imports, and basic AugTab functionality after installation with `pip install augtab`.
- `pyproject.toml` - modern Python project and build-system configuration used for packaging and PyPI distribution.
- `setup.cfg` - package metadata and configuration for installation and distribution.
- `requirements.txt` - Python dependencies required to run AugTab and the accompanying experiments.
- `iSyncTab_Architecture.png` - High-level architecture diagram of the iSyncTab framework.

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
├── AugTab_Architecture.png
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

- `AugTabClassifier` - binary classification
- `AugTabMulti` - multiclass classification
- `AugTabRegressor` - regression

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

- `AugTabClassifier` - binary classification
- `AugTabMulti` - multiclass classification
- `AugTabRegressor` - regression

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

## Example Usage

AugTab supports **binary classification**, **multiclass classification**, and **regression** through three task-specific interfaces:

```python
from augtab import AugTabClassifier, AugTabMulti, AugTabRegressor
```

For a **new dataset**, we recommend using the **Optuna-tuned workflow** because the optimal augmentation width, FAL capacity, MLP backbone size, regularization strengths, learning rate, batch size, and number of training epochs may vary across datasets.

The examples below therefore provide six common configurations:

1. Binary classification without hyperparameter tuning
2. Binary classification with Optuna tuning
3. Multiclass classification without hyperparameter tuning
4. Multiclass classification with Optuna tuning
5. Regression with Optuna tuning
6. Regression without hyperparameter tuning

> **Recommended for new datasets:** Use Examples **2, 4, and 5**, which perform Optuna-based hyperparameter tuning with **5-fold cross-validation**. Keep the test set completely separate from Optuna and use it only for final evaluation.

> **Note:** The examples use synthetic low-dimensional datasets so that they can run without external data files. Replace the synthetic `X` and `y` arrays with your own tabular dataset.

---

### Example 1: Binary Classification Without Hyperparameter Tuning

This example trains `AugTabClassifier` using a fixed configuration.

```python
import random
import numpy as np
import torch

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from augtab import AugTabClassifier, RegularizerConfig


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# ============================================================
# Example binary dataset
# Replace X and y with your own data
# ============================================================

X, y = make_classification(
    n_samples=800,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    random_state=SEED,
)

X = X.astype(np.float32)
y = y.astype(np.int64)


# ============================================================
# Train / test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)


# ============================================================
# Standardization
# Fit preprocessing on training data only
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)


# ============================================================
# Regularization
# ============================================================

regs = RegularizerConfig(
    lambda_sparse=1e-3,
    lambda_div=1e-3,
    lambda_orth=1e-3,
    lambda_budget=1e-3,
    lambda_drift=0.0,
)


# ============================================================
# Initialize AugTab
# ============================================================

model = AugTabClassifier(
    d_features=X_train.shape[1],
    k_aug=32,
    kprime=64,
    h_hidden=64,
    widths=(128, 128),
    activation="gelu",
    append_mask=False,
    gating="basic",
    regs=regs,
    device=DEVICE,
    lr=2e-3,
    weight_decay=1e-4,
)


# ============================================================
# Train
# ============================================================

model.fit(
    X_train,
    y_train,
    epochs=80,
    batch_size=64,
    verbose=False,
)


# ============================================================
# Evaluate
# ============================================================

y_pred = model.predict(X_test).numpy()
y_prob = model.predict_proba(X_test).numpy().reshape(-1)

print("\nBinary Classification Results")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("F1       :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
```

---

### Example 2: Binary Classification with Optuna Hyperparameter Tuning

For a **new binary classification dataset**, this is the recommended workflow.

Optuna searches the AugTab architecture, regularization, and optimization hyperparameters using **5-fold stratified cross-validation** on the training data. The test set remains completely untouched during tuning.

```python
import random
import numpy as np
import optuna
import torch

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from augtab import AugTabClassifier, RegularizerConfig


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# ============================================================
# Example binary dataset
# Replace X and y with your own data
# ============================================================

X, y = make_classification(
    n_samples=800,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    random_state=SEED,
)

X = X.astype(np.float32)
y = y.astype(np.int64)


# ============================================================
# Hold out the final test set BEFORE Optuna
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)


# ============================================================
# Optuna search space
# ============================================================

def sample_params(trial):

    hidden_width = trial.suggest_categorical(
        "hidden_width",
        [64, 128, 256],
    )

    depth = trial.suggest_int(
        "depth",
        1,
        3,
    )

    return {
        "k_aug": trial.suggest_categorical(
            "k_aug",
            [8, 16, 24, 32, 48, 64],
        ),

        "kprime": trial.suggest_categorical(
            "kprime",
            [16, 32, 64, 128],
        ),

        "h_hidden": trial.suggest_categorical(
            "h_hidden",
            [32, 64, 128, 256],
        ),

        "activation": trial.suggest_categorical(
            "activation",
            ["gelu", "relu", "silu"],
        ),

        "widths": tuple(
            [hidden_width] * depth
        ),

        "lambda_sparse": trial.suggest_float(
            "lambda_sparse",
            1e-6,
            1e-2,
            log=True,
        ),

        "lambda_div": trial.suggest_float(
            "lambda_div",
            1e-6,
            1e-2,
            log=True,
        ),

        "lambda_orth": trial.suggest_float(
            "lambda_orth",
            1e-6,
            1e-2,
            log=True,
        ),

        "lambda_budget": trial.suggest_float(
            "lambda_budget",
            1e-6,
            1e-2,
            log=True,
        ),

        "lambda_drift": trial.suggest_categorical(
            "lambda_drift",
            [0.0, 1e-5, 1e-4, 1e-3],
        ),

        "lr": trial.suggest_float(
            "lr",
            1e-4,
            2e-2,
            log=True,
        ),

        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-6,
            3e-3,
            log=True,
        ),

        "batch_size": trial.suggest_categorical(
            "batch_size",
            [32, 64, 128],
        ),

        "epochs": trial.suggest_categorical(
            "epochs",
            [50, 80, 120],
        ),
    }


# ============================================================
# Build AugTab
# ============================================================

def build_model(params, d_features):

    regs = RegularizerConfig(
        lambda_sparse=params["lambda_sparse"],
        lambda_div=params["lambda_div"],
        lambda_orth=params["lambda_orth"],
        lambda_budget=params["lambda_budget"],
        lambda_drift=params["lambda_drift"],
    )

    return AugTabClassifier(
        d_features=d_features,
        k_aug=params["k_aug"],
        kprime=params["kprime"],
        h_hidden=params["h_hidden"],
        widths=params["widths"],
        activation=params["activation"],
        append_mask=False,
        gating="basic",
        regs=regs,
        device=DEVICE,
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )


# ============================================================
# 5-fold Optuna objective
# ============================================================

def objective(trial):

    params = sample_params(trial)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    scores = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train)
    ):

        # ----------------------------------------------------
        # Fold-local preprocessing
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_tr = scaler.fit_transform(
            X_train[train_idx]
        ).astype(np.float32)

        X_val = scaler.transform(
            X_train[val_idx]
        ).astype(np.float32)

        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]


        # ----------------------------------------------------
        # Fresh model for every fold
        # ----------------------------------------------------

        torch.manual_seed(SEED + fold)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + fold)

        model = build_model(
            params,
            d_features=X_tr.shape[1],
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_tr,
            y_tr,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=False,
        )


        # ----------------------------------------------------
        # Validation accuracy
        # ----------------------------------------------------

        score = model.score(
            X_val,
            y_val,
        )

        scores.append(score)


    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    trial.set_user_attr(
        "cv_mean",
        mean_score,
    )

    trial.set_user_attr(
        "cv_std",
        std_score,
    )

    return mean_score


# ============================================================
# Run Optuna
# ============================================================

sampler = optuna.samplers.TPESampler(
    seed=SEED
)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
)


# Small value for demonstration.
# Increase for full experiments.
N_TRIALS = 20

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True,
)


print("\nBest 5-fold CV Accuracy:")
print(
    f"{study.best_trial.user_attrs['cv_mean']:.4f} "
    f"± {study.best_trial.user_attrs['cv_std']:.4f}"
)

print("\nBest Hyperparameters:")
print(study.best_params)


# ============================================================
# Reconstruct widths from Optuna parameters
# ============================================================

best = study.best_params.copy()

best["widths"] = tuple(
    [best["hidden_width"]] * best["depth"]
)


# ============================================================
# Train final model using all training data
# ============================================================

final_scaler = StandardScaler()

X_train_final = final_scaler.fit_transform(
    X_train
).astype(np.float32)

X_test_final = final_scaler.transform(
    X_test
).astype(np.float32)


final_model = build_model(
    best,
    d_features=X_train_final.shape[1],
)


final_model.fit(
    X_train_final,
    y_train,
    epochs=best["epochs"],
    batch_size=best["batch_size"],
    verbose=False,
)


# ============================================================
# Final held-out test evaluation
# ============================================================

y_pred = final_model.predict(
    X_test_final
).numpy()

y_prob = final_model.predict_proba(
    X_test_final
).numpy().reshape(-1)


print("\nFinal Test Results")
print(
    "Accuracy :",
    accuracy_score(y_test, y_pred),
)

print(
    "F1       :",
    f1_score(y_test, y_pred),
)

print(
    "ROC-AUC  :",
    roc_auc_score(y_test, y_prob),
)
```

> **Recommended:** Increase `N_TRIALS` for full experiments. The value above is intentionally kept small so that the example remains practical as a quick-start demonstration.

---

### Example 3: Multiclass Classification Without Hyperparameter Tuning

For multiclass problems, use `AugTabMulti` and provide the number of target classes through `n_classes`.

```python
import random
import numpy as np
import torch

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from augtab import AugTabMulti, RegularizerConfig


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Example multiclass dataset
# ============================================================

X, y = make_classification(
    n_samples=900,
    n_features=15,
    n_informative=10,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=SEED,
)

X = X.astype(np.float32)
y = y.astype(np.int64)

N_CLASSES = len(np.unique(y))


# ============================================================
# Train / test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)


# ============================================================
# Standardization
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
).astype(np.float32)

X_test = scaler.transform(
    X_test
).astype(np.float32)


# ============================================================
# Regularization
# ============================================================

regs = RegularizerConfig(
    lambda_sparse=1e-3,
    lambda_div=1e-3,
    lambda_orth=1e-3,
    lambda_budget=1e-3,
    lambda_drift=0.0,
)


# ============================================================
# Initialize AugTab
# ============================================================

model = AugTabMulti(
    d_features=X_train.shape[1],
    n_classes=N_CLASSES,
    k_aug=32,
    kprime=64,
    h_hidden=64,
    widths=(128, 128),
    activation="gelu",
    append_mask=False,
    gating="basic",
    regs=regs,
    device=DEVICE,
    lr=2e-3,
    weight_decay=1e-4,
)


# ============================================================
# Train
# ============================================================

model.fit(
    X_train,
    y_train,
    epochs=80,
    batch_size=64,
    verbose=False,
)


# ============================================================
# Evaluate
# ============================================================

y_pred = model.predict(
    X_test
).numpy()

y_prob = model.predict_proba(
    X_test
).numpy()


print("\nMulticlass Classification Results")

print(
    "Accuracy :",
    accuracy_score(y_test, y_pred),
)

print(
    "Macro F1 :",
    f1_score(
        y_test,
        y_pred,
        average="macro",
    ),
)

print(
    "Probability matrix shape:",
    y_prob.shape,
)
```

---

### Example 4: Multiclass Classification with Optuna Hyperparameter Tuning

For a **new multiclass dataset**, this is the recommended workflow.

```python
import random
import numpy as np
import optuna
import torch

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from augtab import AugTabMulti, RegularizerConfig


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Example multiclass dataset
# ============================================================

X, y = make_classification(
    n_samples=900,
    n_features=15,
    n_informative=10,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=SEED,
)

X = X.astype(np.float32)
y = y.astype(np.int64)

N_CLASSES = len(np.unique(y))


# ============================================================
# Held-out test set
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)


# ============================================================
# Optuna search space
# ============================================================

def sample_params(trial):

    hidden_width = trial.suggest_categorical(
        "hidden_width",
        [64, 128, 256],
    )

    depth = trial.suggest_int(
        "depth",
        1,
        3,
    )

    return {
        "k_aug": trial.suggest_categorical(
            "k_aug",
            [8, 16, 24, 32, 48, 64],
        ),

        "kprime": trial.suggest_categorical(
            "kprime",
            [16, 32, 64, 128],
        ),

        "h_hidden": trial.suggest_categorical(
            "h_hidden",
            [32, 64, 128, 256],
        ),

        "activation": trial.suggest_categorical(
            "activation",
            ["gelu", "relu", "silu"],
        ),

        "widths": tuple(
            [hidden_width] * depth
        ),

        "lambda_sparse": trial.suggest_float(
            "lambda_sparse",
            1e-6,
            1e-2,
            log=True,
        ),

        "lambda_div": trial.suggest_float(
            "lambda_div",
            1e-6,
            1e-2,
            log=True,
        ),

        "lambda_orth": trial.suggest_float(
            "lambda_orth",
            1e-6,
            1e-2,
            log=True,
        ),

        "lambda_budget": trial.suggest_float(
            "lambda_budget",
            1e-6,
            1e-2,
            log=True,
        ),

        "lambda_drift": trial.suggest_categorical(
            "lambda_drift",
            [0.0, 1e-5, 1e-4, 1e-3],
        ),

        "lr": trial.suggest_float(
            "lr",
            1e-4,
            2e-2,
            log=True,
        ),

        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-6,
            3e-3,
            log=True,
        ),

        "batch_size": trial.suggest_categorical(
            "batch_size",
            [32, 64, 128],
        ),

        "epochs": trial.suggest_categorical(
            "epochs",
            [50, 80, 120],
        ),
    }


# ============================================================
# Build model
# ============================================================

def build_model(params, d_features):

    regs = RegularizerConfig(
        lambda_sparse=params["lambda_sparse"],
        lambda_div=params["lambda_div"],
        lambda_orth=params["lambda_orth"],
        lambda_budget=params["lambda_budget"],
        lambda_drift=params["lambda_drift"],
    )

    return AugTabMulti(
        d_features=d_features,
        n_classes=N_CLASSES,
        k_aug=params["k_aug"],
        kprime=params["kprime"],
        h_hidden=params["h_hidden"],
        widths=params["widths"],
        activation=params["activation"],
        append_mask=False,
        gating="basic",
        regs=regs,
        device=DEVICE,
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )


# ============================================================
# 5-fold Optuna objective
# ============================================================

def objective(trial):

    params = sample_params(trial)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    scores = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train)
    ):

        scaler = StandardScaler()

        X_tr = scaler.fit_transform(
            X_train[train_idx]
        ).astype(np.float32)

        X_val = scaler.transform(
            X_train[val_idx]
        ).astype(np.float32)

        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]


        torch.manual_seed(SEED + fold)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + fold)


        model = build_model(
            params,
            d_features=X_tr.shape[1],
        )


        model.fit(
            X_tr,
            y_tr,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=False,
        )


        score = model.score(
            X_val,
            y_val,
        )

        scores.append(score)


    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    trial.set_user_attr(
        "cv_mean",
        mean_score,
    )

    trial.set_user_attr(
        "cv_std",
        std_score,
    )

    return mean_score


# ============================================================
# Run Optuna
# ============================================================

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(
        seed=SEED
    ),
)


N_TRIALS = 20

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True,
)


print("\nBest 5-fold CV Accuracy:")

print(
    f"{study.best_trial.user_attrs['cv_mean']:.4f} "
    f"± {study.best_trial.user_attrs['cv_std']:.4f}"
)

print("\nBest Hyperparameters:")
print(study.best_params)


# ============================================================
# Final training
# ============================================================

best = study.best_params.copy()

best["widths"] = tuple(
    [best["hidden_width"]] * best["depth"]
)


scaler = StandardScaler()

X_train_final = scaler.fit_transform(
    X_train
).astype(np.float32)

X_test_final = scaler.transform(
    X_test
).astype(np.float32)


final_model = build_model(
    best,
    d_features=X_train_final.shape[1],
)


final_model.fit(
    X_train_final,
    y_train,
    epochs=best["epochs"],
    batch_size=best["batch_size"],
    verbose=False,
)


# ============================================================
# Final test evaluation
# ============================================================

y_pred = final_model.predict(
    X_test_final
).numpy()


print("\nFinal Test Results")

print(
    "Accuracy :",
    accuracy_score(y_test, y_pred),
)

print(
    "Macro F1 :",
    f1_score(
        y_test,
        y_pred,
        average="macro",
    ),
)
```

---

### Example 5: Regression with Optuna Hyperparameter Tuning

For a **new regression dataset**, we recommend tuning `AugTabRegressor` with Optuna and **5-fold cross-validation**.

The optimization objective below maximizes mean cross-validation $R^2$.

```python
import random
import numpy as np
import optuna
import torch

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from augtab import AugTabRegressor, RegularizerConfig


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Example regression dataset
# ============================================================

X, y = make_regression(
    n_samples=800,
    n_features=12,
    n_informative=8,
    noise=15.0,
    random_state=SEED,
)

X = X.astype(np.float32)
y = y.astype(np.float32)


# ============================================================
# Held-out test set
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
)


# ============================================================
# Optuna search space
# ============================================================

def sample_params(trial):

    hidden_width = trial.suggest_categorical(
        "hidden_width",
        [64, 128, 256],
    )

    depth = trial.suggest_int(
        "depth",
        1,
        3,
    )

    return {
        "k_aug": trial.suggest_categorical(
            "k_aug",
            [8, 16, 24, 32, 48, 64],
        ),

        "kprime": trial.suggest_categorical(
            "kprime",
            [16, 32, 64, 128],
        ),

        "h_hidden": trial.suggest_categorical(
            "h_hidden",
            [32, 64, 128, 256],
        ),

        "activation": trial.suggest_categorical(
            "activation",
            ["gelu", "relu", "silu"],
        ),

        "widths": tuple(
            [hidden_width] * depth
        ),

        "lambda_sparse": trial.suggest_float(
            "lambda_sparse",
            1e-6,
            1e-2,
            log=True,
        ),

        "lambda_div": trial.suggest_float(
            "lambda_div",
            1e-6,
            1e-2,
            log=True,
        ),

        "lambda_orth": trial.suggest_float(
            "lambda_orth",
            1e-6,
            1e-2,
            log=True,
        ),

        "lambda_budget": trial.suggest_float(
            "lambda_budget",
            1e-6,
            1e-2,
            log=True,
        ),

        "lambda_drift": trial.suggest_categorical(
            "lambda_drift",
            [0.0, 1e-5, 1e-4, 1e-3],
        ),

        "lr": trial.suggest_float(
            "lr",
            1e-4,
            2e-2,
            log=True,
        ),

        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-6,
            3e-3,
            log=True,
        ),

        "batch_size": trial.suggest_categorical(
            "batch_size",
            [32, 64, 128],
        ),

        "epochs": trial.suggest_categorical(
            "epochs",
            [50, 80, 120],
        ),
    }


# ============================================================
# Build AugTab regressor
# ============================================================

def build_model(params, d_features):

    regs = RegularizerConfig(
        lambda_sparse=params["lambda_sparse"],
        lambda_div=params["lambda_div"],
        lambda_orth=params["lambda_orth"],
        lambda_budget=params["lambda_budget"],
        lambda_drift=params["lambda_drift"],
    )

    return AugTabRegressor(
        d_features=d_features,
        k_aug=params["k_aug"],
        kprime=params["kprime"],
        h_hidden=params["h_hidden"],
        widths=params["widths"],
        activation=params["activation"],
        append_mask=False,
        gating="basic",
        regs=regs,
        device=DEVICE,
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )


# ============================================================
# 5-fold Optuna objective
# ============================================================

def objective(trial):

    params = sample_params(trial)

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    scores = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train)
    ):

        scaler = StandardScaler()

        X_tr = scaler.fit_transform(
            X_train[train_idx]
        ).astype(np.float32)

        X_val = scaler.transform(
            X_train[val_idx]
        ).astype(np.float32)

        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]


        torch.manual_seed(SEED + fold)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + fold)


        model = build_model(
            params,
            d_features=X_tr.shape[1],
        )


        model.fit(
            X_tr,
            y_tr,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=False,
        )


        score = model.score(
            X_val,
            y_val,
        )

        scores.append(score)


    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    trial.set_user_attr(
        "cv_mean",
        mean_score,
    )

    trial.set_user_attr(
        "cv_std",
        std_score,
    )

    return mean_score


# ============================================================
# Run Optuna
# ============================================================

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(
        seed=SEED
    ),
)


N_TRIALS = 20

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True,
)


print("\nBest 5-fold CV R2:")

print(
    f"{study.best_trial.user_attrs['cv_mean']:.4f} "
    f"± {study.best_trial.user_attrs['cv_std']:.4f}"
)

print("\nBest Hyperparameters:")
print(study.best_params)


# ============================================================
# Final training
# ============================================================

best = study.best_params.copy()

best["widths"] = tuple(
    [best["hidden_width"]] * best["depth"]
)


scaler = StandardScaler()

X_train_final = scaler.fit_transform(
    X_train
).astype(np.float32)

X_test_final = scaler.transform(
    X_test
).astype(np.float32)


final_model = build_model(
    best,
    d_features=X_train_final.shape[1],
)


final_model.fit(
    X_train_final,
    y_train,
    epochs=best["epochs"],
    batch_size=best["batch_size"],
    verbose=False,
)


# ============================================================
# Final held-out test evaluation
# ============================================================

y_pred = final_model.predict(
    X_test_final
).numpy().reshape(-1)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred,
    )
)

mae = mean_absolute_error(
    y_test,
    y_pred,
)

r2 = r2_score(
    y_test,
    y_pred,
)


print("\nFinal Test Results")
print("RMSE :", rmse)
print("MAE  :", mae)
print("R2   :", r2)
```

---

### Example 6: Regression Without Hyperparameter Tuning

This example trains `AugTabRegressor` using a fixed configuration without Optuna.

```python
import random
import numpy as np
import torch

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from augtab import AugTabRegressor, RegularizerConfig


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Example regression dataset
# ============================================================

X, y = make_regression(
    n_samples=800,
    n_features=12,
    n_informative=8,
    noise=15.0,
    random_state=SEED,
)

X = X.astype(np.float32)
y = y.astype(np.float32)


# ============================================================
# Train / test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
)


# ============================================================
# Standardization
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
).astype(np.float32)

X_test = scaler.transform(
    X_test
).astype(np.float32)


# ============================================================
# Regularization
# ============================================================

regs = RegularizerConfig(
    lambda_sparse=1e-3,
    lambda_div=1e-3,
    lambda_orth=1e-3,
    lambda_budget=1e-3,
    lambda_drift=0.0,
)


# ============================================================
# Initialize AugTab
# ============================================================

model = AugTabRegressor(
    d_features=X_train.shape[1],
    k_aug=32,
    kprime=64,
    h_hidden=64,
    widths=(128, 128),
    activation="gelu",
    append_mask=False,
    gating="basic",
    regs=regs,
    device=DEVICE,
    lr=2e-3,
    weight_decay=1e-4,
)


# ============================================================
# Train
# ============================================================

model.fit(
    X_train,
    y_train,
    epochs=80,
    batch_size=64,
    verbose=False,
)


# ============================================================
# Evaluate
# ============================================================

y_pred = model.predict(
    X_test
).numpy().reshape(-1)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred,
    )
)

mae = mean_absolute_error(
    y_test,
    y_pred,
)

r2 = r2_score(
    y_test,
    y_pred,
)


print("\nRegression Results")
print("RMSE :", rmse)
print("MAE  :", mae)
print("R2   :", r2)
```

---

### Adapting AugTab to Your Own Dataset

To apply AugTab to a new low-dimensional tabular dataset:

1. Replace the synthetic `X` and `y` arrays with your own feature matrix and target.
2. Encode categorical variables before passing them to AugTab.
3. Encode classification labels as integer class IDs.
4. Standardize numerical features using statistics computed from the training split only.
5. Set `d_features=X.shape[1]` after preprocessing.
6. Use `AugTabClassifier` for binary classification.
7. Use `AugTabMulti` for multiclass classification and specify `n_classes`.
8. Use `AugTabRegressor` for regression.
9. Use `append_mask=False` when no explicit missingness mask is required.
10. For a new dataset, prefer the **Optuna workflow** to select AugTab and optimization hyperparameters.
11. During Optuna tuning, perform preprocessing independently inside each cross-validation fold.
12. Initialize a **fresh AugTab model for every cross-validation fold**.
13. Keep the test set completely separate from hyperparameter tuning and use it only for final evaluation.

> **Recommended for new datasets:** The fixed-configuration examples are useful for quick experiments and installation checks, while the **Optuna + 5-fold cross-validation workflows** are recommended when reporting dataset-specific AugTab results.

## Related Work and Project Context

AugTab is part of my broader work on tabular deep learning, but it was developed as a **separate collaborative project outside my PhD dissertation research**. Like **ZAYAN**, it explores a complementary direction beyond my primary PhD research on feature ordering, sequencing, and high-dimensional tabular learning.

For broader context, some of our related tabular and multimodal learning projects are listed below.

### GOTabPFN - ICML 2026

**GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data**

- **Venue:** International Conference on Machine Learning (ICML 2026)
- **GitHub:** https://github.com/zadid6pretam/GOTabPFN

### iSyncTab - ECCV 2026

**iSyncTab: Learning Cross-Modal Feature Sequencing for Image-Tabular Data via Neural Synchrony**

- **Venue:** European Conference on Computer Vision (ECCV 2026)
- **GitHub:** https://github.com/zadid6pretam/iSyncTab

### BSTabDiff - ICLR 2026 DeLTa Workshop

**BSTabDiff: Block-Subunit Diffusion Priors for High-Dimensional Tabular Data Generation**

- **Venue:** ICLR 2026 Workshop on Deep Generative Models in Machine Learning: Theory, Principle and Efficacy (DeLTa)
- **GitHub:** https://github.com/zadid6pretam/BSTabDiff

### iStructTab - ICPR 2026

**iStructTab: Structured Feature Sequencing for Multimodal Learning of Image and Tabular Data**

- **Venue:** International Conference on Pattern Recognition (ICPR 2026)
- **GitHub:** https://github.com/zadid6pretam/iStructTab

### DynaTab - AAAI 2026 NeurAI Workshop

**DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data**

- **Venue:** AAAI 2026 Workshop on NeuroAI Multimodal Intelligence
- **GitHub:** https://github.com/zadid6pretam/DynaTab

### TabSeq - ICPR 2024

**TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering**

- **Venue:** International Conference on Pattern Recognition (ICPR 2024)
- **GitHub:** https://github.com/zadid6pretam/TabSeq

---

### ZAYAN - ICPR 2026

**ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing Data**

- **Venue:** International Conference on Pattern Recognition (ICPR 2026)
- **GitHub:** https://github.com/zadid6pretam/ZAYAN

> **Note:** AugTab and ZAYAN were developed as separate collaborative projects outside my PhD dissertation research.

## Contact

For any questions, issues, or suggestions related to AugTab, please feel free to open an issue on GitHub.
