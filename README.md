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

For an input feature vector \(x\), AugTab preserves the original features and concatenates them with the gated learned augmentation:

\[
\tilde{x} = x \oplus \big(g(x) \odot z(x)\big),
\]

where \(z(x)\) represents the candidate augmented features and \(g(x)\) provides input-dependent gates controlling their activation.

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

This repository is provided for the review process and currently includes:

- `AugTab.py` — core implementation of the AugTab model
- `AugTab Try.ipynb` — example notebook demonstrating how to use AugTab with **Optuna-based hyperparameter tuning** and **5-fold cross-validation**, reporting **mean accuracy ± standard deviation** on the **Water Potability** dataset

The notebook is written so that the dataset file and target column can be changed easily, allowing the same workflow to be reused for other tabular datasets.

In addition, the notebook includes supplementary diagnostic and analysis code used to further inspect AugTab’s behavior.

---

## Repository Contents

```text
.
├── AugTab.py
├── AugTab Try.ipynb
├── requirements.txt
└── README.md
