"""AugTab: Learnable Feature Augmentation for Low-Dimensional Tabular Data

Core implementation of the AugTab framework.


Public estimators:
  • AugTabClassifier  (binary classification)
  • AugTabMulti       (multiclass classification)
  • AugTabRegressor   (regression)


Under the hood, all estimators use the same core module
(Feature Augmentation Layer + MLP backbone) and expose a sklearn-like API:


    fit(X, y, M=None, ...)
    predict(X, M=None)
    predict_proba(X, M=None)  # classifiers only
    score(X, y, M=None)       # accuracy (classification) or R² (regression)


Implements the paper-spec architecture with:
  • Feature Augmentation Layer (FAL):
      cross ⊙ branch + tiny-MLP branch → projection P → gated by g(x)
  • Optional uncertainty-aware gating using a lightweight probe or EMA teacher head
  • Robustness add-ons:
      drift-consistency loss and missingness-aware input ([x∘m ; m])
  • Regularizers:
      sparsity (‖g‖₁), diversity (offdiag(Cov(Z))),
      orthogonality (‖XᵀZ‖_F²), and budget penalty
  • Backbone: compact MLP
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Literal, List


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


Tensor = torch.Tensor


# ------------------------------
# Utilities
# ------------------------------


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")




def offdiag_fro2(mat: Tensor) -> Tensor:
    """‖offdiag(mat)‖_F^2 for square matrices."""
    m = mat
    return (m.square().sum(dim=(-2, -1)) - m.diagonal(dim1=-2, dim2=-1).square().sum(dim=-1))




def cov_batch(Z: Tensor) -> Tensor:
    """Covariance of Z along batch dimension: Z ∈ [B, k] → Cov ∈ [k, k]."""
    B = Z.shape[0]
    if B < 2:
        return Z.new_zeros(Z.shape[1], Z.shape[1])
    Zc = Z - Z.mean(dim=0, keepdim=True)
    return (Zc.T @ Zc) / (B - 1)




def fro2(mat: Tensor) -> Tensor:
    return (mat.square()).sum()




def ensure_tensor(x, device, dtype=None) -> Tensor:
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif torch.is_tensor(x):
        t = x
    else:
        t = torch.tensor(x)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(device)




# ------------------------------
# Configs
# ------------------------------
@dataclass
class FALConfig:
    d_in: int                    # expected input dim to FAL (after optional mask concatenation)
    kprime: int = 64             # width before projection per branch
    k_aug: int = 64              # final augmented width k
    h_hidden: int = 128          # hidden width for the MLP branch
    activation: str = "gelu"
    dropout: float = 0.0


    # Gating
    gating: Literal["basic", "uncertainty"] = "basic"
    tau: float = 0.5
    alpha: float = 1.0           # scale for (tau - margin)


    # Uncertainty margin source
    margin_source: Literal["probe", "ema"] = "probe"
    probe_hidden: int = 64       # r for the 1–2 layer probe
    probe_layers: int = 1        # 1 or 2


    # Initialization
    init: Literal["kaiming", "xavier"] = "kaiming"




@dataclass
class BackboneConfig:
    d_in: int
    n_classes: int = 1           # C; for regression/binary this can be 1
    task: Literal["binary", "multiclass", "regression"] = "multiclass"


    widths: Tuple[int, ...] = (128, 128)
    activation: str = "gelu"
    dropout: float = 0.0
    batchnorm: bool = False




@dataclass
class RegularizerConfig:
    lambda_sparse: float = 1e-3
    lambda_div: float = 1e-3
    lambda_orth: float = 1e-3
    lambda_budget: float = 1e-3
    lambda_drift: float = 0.0         # turn on to enforce drift-consistency


    beta0: float = 0.0                # constant term in budget penalty
    budget_betas: Optional[Tensor] = None  # if None → ones(k)


    # Drift settings
    drift_kind: Literal["additive", "multiplicative"] = "additive"
    drift_sigma: float = 0.05         # relative scale




@dataclass
class AugTabConfig:
    # Base data dims
    d_features: int
    append_mask: bool = True          # if True, expect mask m, and FAL sees [x∘m ; m]


    # FAL + Backbone
    fal: FALConfig = field(default_factory=lambda: FALConfig(d_in=1))
    backbone: BackboneConfig = field(default_factory=lambda: BackboneConfig(d_in=1))


    regs: RegularizerConfig = field(default_factory=RegularizerConfig)


    # Misc
    use_logits_temperature: bool = False
    temperature: float = 1.0


    def finalize_dims(self):
        """Set inner dims based on switches.
        If append_mask=True, FAL input dim = d_features * 2, else = d_features.
        Backbone input dim = d_features + k_aug.
        """
        fal_in = self.d_features * 2 if self.append_mask else self.d_features
        self.fal.d_in = fal_in
        self.backbone.d_in = self.d_features + self.fal.k_aug




# ------------------------------
# Modules (FAL + Backbone)
# ------------------------------
class TinyProbe(nn.Module):
    """Lightweight probe for uncertainty margin: softmax over C.
    Supports 1–2 layers: d → r → C.
    """
    def __init__(self, d_in: int, n_classes: int, hidden: int = 64, layers: int = 1, act: str = "gelu"):
        super().__init__()
        self.layers = layers
        self.act = get_activation(act)
        if layers == 1:
            self.lin = nn.Linear(d_in, n_classes)
        elif layers == 2:
            self.lin1 = nn.Linear(d_in, hidden)
            self.lin2 = nn.Linear(hidden, n_classes)
        else:
            raise ValueError("probe_layers must be 1 or 2")


    @torch.no_grad()
    def margin(self, x: Tensor) -> Tensor:
        if self.layers == 1:
            logits = self.lin(x)
        else:
            logits = self.lin2(F.gelu(self.lin1(x)))
        probs = logits.softmax(dim=-1)
        top2 = probs.topk(k=min(2, probs.size(-1)), dim=-1).values
        if probs.size(-1) == 1:
            m = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        else:
            m = (top2[:, 0] - top2[:, 1])
        return m.unsqueeze(-1)




class EMATeacher(nn.Module):
    def __init__(self, probe: TinyProbe, decay: float = 0.99):
        super().__init__()
        self.decay = decay
        import copy
        self.teacher = copy.deepcopy(probe)
        for p in self.teacher.parameters():
            p.requires_grad_(False)


    @torch.no_grad()
    def update(self, student: TinyProbe):
        for p_t, p_s in zip(self.teacher.parameters(), student.parameters()):
            p_t.data.mul_((self.decay)).add_(p_s.data, alpha=(1.0 - self.decay))


    @torch.no_grad()
    def margin(self, x: Tensor) -> Tensor:
        return self.teacher.margin(x)




# -------- Safe BatchNorm (prevents crash on batch size 1) --------
class SafeBatchNorm1d(nn.BatchNorm1d):
    """BN that gracefully handles batch size 1 by using running stats."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) == 1:
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias,
                False,  # training=False -> use running stats
                self.momentum, self.eps
            )
        return super().forward(x)




class FAL(nn.Module):
    def __init__(self, cfg: FALConfig, d_features: int, append_mask: bool, n_classes: int):
        super().__init__()
        self.cfg = cfg
        self.d_features = d_features
        self.append_mask = append_mask


        act = get_activation(cfg.activation)
        self.act = act
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()


        d_in = cfg.d_in
        kprime = cfg.kprime
        k = cfg.k_aug


        # Branch: tiny MLP → k'
        self.W1 = nn.Linear(d_in, cfg.h_hidden)
        self.W2 = nn.Linear(cfg.h_hidden, kprime)


        # Branch: cross (Ax) ⊙ (Bx) → k'
        self.A = nn.Linear(d_in, kprime, bias=False)
        self.B = nn.Linear(d_in, kprime, bias=False)


        # Projection P: 2k' → k
        self.P = nn.Linear(2 * kprime, k)


        # Gate g(x)
        self.Wg = nn.Linear(d_in, k)


        # Probe / teacher for uncertainty-aware gating
        if cfg.gating == "uncertainty":
            self.probe = TinyProbe(d_in=d_features if not append_mask else (d_features * 2),
                                   n_classes=max(2, n_classes),
                                   hidden=cfg.probe_hidden,
                                   layers=cfg.probe_layers,
                                   act=cfg.activation)
            self.ema_teacher = None
            if cfg.margin_source == "ema":
                self.ema_teacher = EMATeacher(self.probe, decay=0.99)
        else:
            self.probe = None
            self.ema_teacher = None


        self.reset_parameters()


    def reset_parameters(self):
        if self.cfg.init == "kaiming":
            for m in [self.W1, self.W2, self.A, self.B, self.P, self.Wg]:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(m.bias, -bound, bound)
        else:
            for m in [self.W1, self.W2, self.A, self.B, self.P, self.Wg]:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


    def _prepare_input(self, x: Tensor, m: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if m is not None:
            x_core = x * m
            x_in = torch.cat([x_core, m], dim=-1) if self.append_mask else x_core
        else:
            x_core = x
            x_in = x
        return x_in, x_core


    @torch.no_grad()
    def update_ema(self):
        if self.ema_teacher is not None:
            self.ema_teacher.update(self.probe)


    def forward(self, x: Tensor, m: Optional[Tensor] = None) -> Dict[str, Tensor]:
        x_in, x_core = self._prepare_input(x, m)


        u = self.W1(x_in)
        h = self.act(u)
        h = self.drop(h)
        z_mlp = self.W2(h)


        z_cross = self.A(x_in) * self.B(x_in)


        z = self.P(torch.cat([z_mlp, z_cross], dim=-1))


        g_pre = self.Wg(x_in)
        if self.cfg.gating == "uncertainty":
            with torch.no_grad():
                margin = (self.ema_teacher.margin(x_in) if (self.cfg.margin_source == "ema" and self.ema_teacher is not None)
                          else self.probe.margin(x_in))
            g_pre = g_pre + self.cfg.alpha * (self.cfg.tau - margin)


        g = torch.sigmoid(g_pre)
        aug = g * z
        tilde_x = torch.cat([x_core, aug], dim=-1)


        return {"tilde_x": tilde_x, "z": z, "g": g, "z_mlp": z_mlp, "z_cross": z_cross, "x_in": x_in, "x_core": x_core}




class MLPBackbone(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        dims: List[int] = [cfg.d_in] + list(cfg.widths) + [cfg.n_classes]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if cfg.batchnorm:
                layers.append(SafeBatchNorm1d(dims[i + 1]))  # safe BN
            layers.append(get_activation(cfg.activation))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)




# ------------------------------
# Core model (task-agnostic)
# ------------------------------
class AugTabCore(nn.Module):
    def __init__(self, cfg: AugTabConfig):
        super().__init__()
        self.cfg = cfg
        cfg.finalize_dims()


        self.fal = FAL(cfg.fal, d_features=cfg.d_features, append_mask=cfg.append_mask,
                       n_classes=cfg.backbone.n_classes if cfg.backbone.task != "binary" else 2)
        self.backbone = MLPBackbone(cfg.backbone)


        # Loss per task
        if cfg.backbone.task == "multiclass":
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.backbone.task == "binary":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()


        # Budget betas
        k = cfg.fal.k_aug
        if cfg.regs.budget_betas is None:
            self.register_buffer("budget_betas", torch.ones(k))
        else:
            bb = cfg.regs.budget_betas
            if isinstance(bb, Tensor):
                assert bb.numel() == k, "budget_betas must have k elements"
                self.register_buffer("budget_betas", bb.reshape(-1).clone())
            else:
                raise TypeError("budget_betas must be a torch.Tensor or None")


    @torch.no_grad()
    def update_ema(self):
        self.fal.update_ema()


    def forward(self, x: Tensor, m: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        fal_out = self.fal(x, m)
        logits = self.backbone(fal_out["tilde_x"])  # [B, C] or [B, 1]
        if self.cfg.use_logits_temperature and self.cfg.temperature != 1.0 and self.cfg.backbone.task != "binary":
            logits = logits / self.cfg.temperature
        return logits, fal_out


    # ------------------------------ Regularizers ------------------------------
    def reg_losses(self, X: Tensor, Z: Tensor, G: Tensor, fal_x_in: Tensor) -> Dict[str, Tensor]:
        regs = self.cfg.regs
        out: Dict[str, Tensor] = {}
        L_sparse = G.abs().sum(dim=1).mean()
        out["sparse"] = regs.lambda_sparse * L_sparse
        CovZ = cov_batch(Z)
        out["div"] = regs.lambda_div * offdiag_fro2(CovZ)
        out["orth"] = regs.lambda_orth * fro2(X.T @ Z)
        mean_g = G.mean(dim=0)
        L_budget = regs.beta0 + (self.budget_betas * mean_g).sum()
        out["budget"] = regs.lambda_budget * L_budget
        return out


    # ------------------------------ Drift consistency ------------------------------
    def sample_drift(self, X_core: Tensor) -> Tensor:
        regs = self.cfg.regs
        if regs.drift_kind == "additive":
            std = torch.clamp(X_core.std(dim=0, keepdim=True), min=1e-6)
            return torch.randn_like(X_core) * (regs.drift_sigma * std)
        else:
            eps = torch.randn_like(X_core) * regs.drift_sigma
            return X_core * eps


    def drift_loss(self, x: Tensor, m: Optional[Tensor], z_ref: Tensor) -> Tensor:
        if self.cfg.regs.lambda_drift <= 0:
            return x.new_tensor(0.0)
        fal_in, x_core = self.fal._prepare_input(x, m)
        delta = self.sample_drift(x_core)
        x_pert = x_core + delta
        if m is not None:
            x_pert_in = torch.cat([x_pert, m], dim=-1) if self.cfg.append_mask else x_pert
        else:
            x_pert_in = x_pert
        u = self.fal.W1(x_pert_in)
        h = self.fal.act(u)
        h = self.fal.drop(h)
        z_mlp_p = self.fal.W2(h)
        z_cross_p = self.fal.A(x_pert_in) * self.fal.B(x_pert_in)
        z_p = self.fal.P(torch.cat([z_mlp_p, z_cross_p], dim=-1))
        return F.mse_loss(z_p, z_ref, reduction="mean")


    # ------------------------------ Loss wrapper ------------------------------
    def compute_loss(self, x: Tensor, y: Tensor, m: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        logits, aux = self.forward(x, m)
        task = self.cfg.backbone.task
        if task == "multiclass":
            loss_task = self.criterion(logits, y.long())
        elif task == "binary":
            yf = y.float().view(-1, 1)
            loss_task = self.criterion(logits.view(-1, 1), yf)
        else:  # regression
            yv = y.view_as(logits)
            loss_task = self.criterion(logits, yv)


        regs = self.reg_losses(X=aux["x_core"], Z=aux["z"], G=aux["g"], fal_x_in=aux["x_in"])  # noqa
        loss_drift = self.cfg.regs.lambda_drift * self.drift_loss(x, m, aux["z"]) if self.cfg.regs.lambda_drift > 0 else x.new_tensor(0.0)


        total = loss_task + sum(regs.values()) + loss_drift
        details = {"total": total.detach(), "task": loss_task.detach(), **{f"reg_{k}": v.detach() for k, v in regs.items()}, "drift": loss_drift.detach()}
        return total, details


    # ------------------------------ Inference helpers ------------------------------
    @torch.no_grad()
    def predict_proba(self, x: Tensor, m: Optional[Tensor] = None) -> Tensor:
        logits, _ = self.forward(x, m)
        if self.cfg.backbone.task == "multiclass":
            return logits.softmax(dim=-1)
        if self.cfg.backbone.task == "binary":
            return torch.sigmoid(logits.view(-1, 1))
        raise ValueError("predict_proba is only for classification tasks")


    @torch.no_grad()
    def predict(self, x: Tensor, m: Optional[Tensor] = None) -> Tensor:
        task = self.cfg.backbone.task
        logits, _ = self.forward(x, m)
        if task == "multiclass":
            return logits.argmax(dim=1)
        if task == "binary":
            probs = torch.sigmoid(logits.view(-1, 1)).squeeze(1)
            return (probs >= 0.5).long()
        return logits  # regression




# ------------------------------
# Datasets & training utilities
# ------------------------------
class TabDataset(Dataset):
    def __init__(self, X: Tensor, y: Optional[Tensor] = None, M: Optional[Tensor] = None):
        self.X, self.y, self.M = X, y, M
    def __len__(self):
        return self.X.size(0)
    def __getitem__(self, idx):
        x = self.X[idx]
        m = None if self.M is None else self.M[idx]
        if self.y is None:
            return x, m
        return x, self.y[idx], m




def _accuracy(logits: Tensor, y: Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())




def _accuracy_binary(logits: Tensor, y: Tensor) -> float:
    probs = torch.sigmoid(logits.view(-1, 1)).squeeze(1)
    preds = (probs >= 0.5).long()
    return float((preds == y.long()).float().mean().item())




def _r2(pred: Tensor, y: Tensor) -> float:
    y_mean = y.mean()
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum() + 1e-12
    return float(1.0 - (ss_res / ss_tot))




# ------------------------------
# High-level wrappers (sklearn-like)
# ------------------------------
class _AugTabBase:
    def __init__(self, cfg: AugTabConfig, device: Optional[str] = None, lr: float = 2e-3, weight_decay: float = 1e-4):
        self.cfg = cfg
        self.model = AugTabCore(cfg)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)


    # --------------- public API ---------------
    def fit(self, X, y, M=None, X_val=None, y_val=None, M_val=None,
            epochs: int = 50, batch_size: int = 256, eval_every: int = 1, patience: Optional[int] = None, verbose: bool = True):
        X = ensure_tensor(X, self.device, torch.float32)
        y = ensure_tensor(y, self.device)
        if M is None:
            M = torch.ones_like(X)
        else:
            M = ensure_tensor(M, self.device, torch.float32)


        train_ds = TabDataset(X, y, M)
        # keep drop_last=False (SafeBatchNorm1d handles batch size 1 robustly)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)


        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = ensure_tensor(X_val, self.device, torch.float32)
            y_val = ensure_tensor(y_val, self.device)
            M_val = torch.ones_like(X_val) if M_val is None else ensure_tensor(M_val, self.device, torch.float32)


        best_metric = -float("inf")
        patience_left = patience


        for epoch in range(1, epochs + 1):
            self.model.train()
            for xb, yb, mb in train_dl:
                loss, _ = self.model.compute_loss(xb, yb, mb)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
            self.model.update_ema()


            if verbose and (epoch % eval_every == 0 or epoch == epochs):
                train_metric = self._metric_on(X, y, M)
                msg = f"epoch {epoch:03d} | train {self.metric_name}: {train_metric:.4f}"
                if has_val:
                    val_metric = self._metric_on(X_val, y_val, M_val)
                    msg += f" | val {self.metric_name}: {val_metric:.4f}"
                    if val_metric > best_metric:
                        best_metric = val_metric
                        patience_left = patience
                    elif patience is not None:
                        patience_left -= 1
                        if patience_left <= 0:
                            if verbose:
                                print(msg + " | early stop")
                            break
                if verbose:
                    print(msg)
        return self


    @torch.no_grad()
    def predict(self, X, M=None):
        X = ensure_tensor(X, self.device, torch.float32)
        M = torch.ones_like(X) if M is None else ensure_tensor(M, self.device, torch.float32)
        was_training = self.model.training
        self.model.eval()
        try:
            out = self.model.predict(X, M).detach().cpu()
        finally:
            self.model.train(was_training)
        return out


    @torch.no_grad()
    def predict_proba(self, X, M=None):
        X = ensure_tensor(X, self.device, torch.float32)
        M = torch.ones_like(X) if M is None else ensure_tensor(M, self.device, torch.float32)
        was_training = self.model.training
        self.model.eval()
        try:
            out = self.model.predict_proba(X, M).detach().cpu()
        finally:
            self.model.train(was_training)
        return out


    @torch.no_grad()
    def score(self, X, y, M=None) -> float:
        X = ensure_tensor(X, self.device, torch.float32)
        y = ensure_tensor(y, self.device)
        M = torch.ones_like(X) if M is None else ensure_tensor(M, self.device, torch.float32)
        return self._metric_on(X, y, M)


    # --------------- internal ---------------
    @torch.no_grad()
    def _metric_on(self, X: Tensor, y: Tensor, M: Tensor) -> float:
        was_training = self.model.training
        self.model.eval()
        try:
            logits, _ = self.model(X, M)
            task = self.model.cfg.backbone.task
            if task == "multiclass":
                return _accuracy(logits, y)
            if task == "binary":
                return _accuracy_binary(logits, y)
            return _r2(logits, y.view_as(logits))
        finally:
            self.model.train(was_training)


    def to(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self




# ------------------------------
# Public task-specific classes
# ------------------------------
class AugTabClassifier(_AugTabBase):
    """Binary classifier (BCEWithLogits), exposes predict_proba (Nx1)."""
    def __init__(self, d_features: int, k_aug: int = 32, kprime: int = 64, h_hidden: int = 64,
                 widths: Tuple[int, ...] = (128, 128), activation: str = "gelu",
                 append_mask: bool = True, gating: str = "basic",
                 regs: Optional[RegularizerConfig] = None,
                 device: Optional[str] = None, lr: float = 2e-3, weight_decay: float = 1e-4):
        regs = regs or RegularizerConfig()
        cfg = AugTabConfig(
            d_features=d_features,
            append_mask=append_mask,
            fal=FALConfig(d_in=1, kprime=kprime, k_aug=k_aug, h_hidden=h_hidden, activation=activation, gating=gating),
            backbone=BackboneConfig(d_in=1, n_classes=1, task="binary", widths=widths, activation=activation),
            regs=regs,
        )
        cfg.finalize_dims()
        super().__init__(cfg, device=device, lr=lr, weight_decay=weight_decay)
        self.metric_name = "acc"




class AugTabMulti(_AugTabBase):
    """Multiclass classifier (CrossEntropy), exposes predict_proba (NxC)."""
    def __init__(self, d_features: int, n_classes: int, k_aug: int = 32, kprime: int = 64, h_hidden: int = 64,
                 widths: Tuple[int, ...] = (128, 128), activation: str = "gelu",
                 append_mask: bool = True, gating: str = "basic",
                 regs: Optional[RegularizerConfig] = None,
                 device: Optional[str] = None, lr: float = 2e-3, weight_decay: float = 1e-4):
        regs = regs or RegularizerConfig()
        cfg = AugTabConfig(
            d_features=d_features,
            append_mask=append_mask,
            fal=FALConfig(d_in=1, kprime=kprime, k_aug=k_aug, h_hidden=h_hidden, activation=activation, gating=gating),
            backbone=BackboneConfig(d_in=1, n_classes=n_classes, task="multiclass", widths=widths, activation=activation),
            regs=regs,
        )
        cfg.finalize_dims()
        super().__init__(cfg, device=device, lr=lr, weight_decay=weight_decay)
        self.metric_name = "acc"




class AugTabRegressor(_AugTabBase):
    """Regressor (MSE), score() returns R^2."""
    def __init__(self, d_features: int, k_aug: int = 32, kprime: int = 64, h_hidden: int = 64,
                 widths: Tuple[int, ...] = (128, 128), activation: str = "gelu",
                 append_mask: bool = True, gating: str = "basic",
                 regs: Optional[RegularizerConfig] = None,
                 device: Optional[str] = None, lr: float = 2e-3, weight_decay: float = 1e-4):
        regs = regs or RegularizerConfig()
        cfg = AugTabConfig(
            d_features=d_features,
            append_mask=append_mask,
            fal=FALConfig(d_in=1, kprime=kprime, k_aug=k_aug, h_hidden=h_hidden, activation=activation, gating=gating),
            backbone=BackboneConfig(d_in=1, n_classes=1, task="regression", widths=widths, activation=activation),
            regs=regs,
        )
        cfg.finalize_dims()
        super().__init__(cfg, device=device, lr=lr, weight_decay=weight_decay)
        self.metric_name = "R2"




__all__ = [
    "FALConfig", "BackboneConfig", "RegularizerConfig", "AugTabConfig",
    "AugTabCore", "AugTabClassifier", "AugTabMulti", "AugTabRegressor",
]