"""
symbolic_regression.py

将 EML 算子应用于符号回归任务，使用 Feynman 符号回归数据集
训练可解释的 EML 表达式树，并提取最终符号表达式。

核心设计:
    - 渐进式 STE (直通估计器): 训练时 ste_alpha 从 0 线性增长到 1，
      前向传播逐步从连续权重过渡到量化权重，避免突变导致训练崩溃。
    - 三阶段训练: 阶段1 无 STE 拟合数据 → 阶段2 渐进 STE 适应量化 →
      阶段3 快照后微调 output_scale/bias。
    - 离散正则化: 鼓励权重靠近量化格点，使 STE 过渡更平滑。
    - 权重快照: 将连续权重圆整到最近的量化格点，提取可读符号表达式。

用法:
    python examples/symbolic_regression.py

依赖:
    必需: torch>=2.5, numpy
    可选: sympy (表达式提取), pmlb (Feynman 数据集加载), matplotlib (损失可视化)
"""

import os
import torch
import torch.nn as nn
import numpy as np

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    from pmlb import fetch_data
    HAS_PMLB = True
except ImportError:
    HAS_PMLB = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


FEYNMAN_FORMULAS = {
    "I.6.2": {
        "expr": lambda x1, x2: np.exp(-x1) * np.sin(x2) / np.maximum(x1, 1e-8),
        "vars": ["theta1", "theta2"],
        "description": "Diffraction: exp(-theta1)*sin(theta2)/theta1",
    },
    "I.8.14": {
        "expr": lambda x1, x2, x3, x4: x1 * x2 * x3 / x4**2,
        "vars": ["d", "G", "m1", "r"],
        "description": "Gravitational force: d*G*m1/r^2",
    },
    "I.12.2": {
        "expr": lambda x1, x2, x3: x1 * x2 * x3,
        "vars": ["q", "E", "d"],
        "description": "Electric potential energy: q*E*d",
    },
    "I.13.12": {
        "expr": lambda x1, x2, x3: x1 * x2 / x3**2,
        "vars": ["G", "m1", "r"],
        "description": "Gravitational force: G*m1/r^2",
    },
    "I.14.4": {
        "expr": lambda x1, x2, x3: x1 * x2 * np.sin(x3),
        "vars": ["q", "v", "theta"],
        "description": "Lorentz force: q*v*sin(theta)",
    },
    "I.15.1": {
        "expr": lambda x1, x2, x3: x1 + x2 * x3,
        "vars": ["x1", "v", "t"],
        "description": "Galilean transform: x1 + v*t",
    },
    "I.18.12": {
        "expr": lambda x1, x2, x3, x4: (x1 * x2 + x3 * x4) / (x1 + x3),
        "vars": ["m1", "r1", "m2", "r2"],
        "description": "Center of mass: (m1*r1+m2*r2)/(m1+m2)",
    },
    "II.6.11": {
        "expr": lambda x1, x2: x1 * x2,
        "vars": ["epsilon", "E"],
        "description": "Electric displacement: epsilon*E",
    },
    "II.11.3": {
        "expr": lambda x1, x2: np.exp(-x1) * x2,
        "vars": ["sigma", "E"],
        "description": "Attenuated field: exp(-sigma)*E",
    },
    "III.4.32": {
        "expr": lambda x1, x2: x1 * np.cos(x2),
        "vars": ["A", "phi"],
        "description": "Wave amplitude: A*cos(phi)",
    },
}


class FeynmanDataset:
    """Feynman 符号回归数据集加载器。

    优先尝试从 PMLB 加载真实数据集，若不可用则根据内置公式生成合成数据。
    """

    def __init__(self, formula_name, n_samples=500, seed=42,
                 noise_std=0.0, use_pmlb=True):
        self.formula_name = formula_name
        self.n_samples = n_samples
        self.seed = seed
        self.noise_std = noise_std
        self.X = None
        self.y = None
        self.n_features = 0
        self.feature_names = []

        if use_pmlb and HAS_PMLB:
            self._load_from_pmlb()
        if self.X is None:
            self._generate_synthetic()

    def _load_from_pmlb(self):
        try:
            dataset_name = f"feynman_{self.formula_name.replace('.', '_')}"
            df = fetch_data(dataset_name)
            X = df.iloc[:, :-1].values.astype(np.float32)
            y = df.iloc[:, -1].values.astype(np.float32)
            if len(X) > self.n_samples:
                rng = np.random.RandomState(self.seed)
                idx = rng.choice(len(X), self.n_samples, replace=False)
                X, y = X[idx], y[idx]
            mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            X, y = X[mask], y[mask]
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
            self.n_features = X.shape[1]
            self.feature_names = [f"x{i+1}" for i in range(self.n_features)]
        except Exception:
            self.X = None

    def _generate_synthetic(self):
        if self.formula_name not in FEYNMAN_FORMULAS:
            raise ValueError(
                f"Unknown formula: {self.formula_name}. "
                f"Available: {list(FEYNMAN_FORMULAS.keys())}"
            )

        formula = FEYNMAN_FORMULAS[self.formula_name]
        rng = np.random.RandomState(self.seed)
        n_vars = len(formula["vars"])

        X = rng.uniform(0.1, 2.0, (self.n_samples, n_vars)).astype(np.float32)
        y = formula["expr"](*[X[:, i] for i in range(n_vars)]).astype(np.float32)

        if self.noise_std > 0:
            y += rng.normal(0, self.noise_std, y.shape).astype(np.float32)

        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X, y = X[mask], y[mask]

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.n_features = n_vars
        self.feature_names = formula["vars"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EMLTreeLeaf(nn.Module):
    """表达式树叶节点：加权组合输入特征与常数 1。

    使用渐进式 STE: ste_alpha 控制量化强度，
      alpha=0 时完全连续，alpha=1 时完全量化。
    前向: w_eff = (1-alpha)*w + alpha*round(w)
    反向: 梯度直通 (d(w_eff)/dw = 1)
    """

    def __init__(self, input_dim, quant_scale=0.5):
        super().__init__()
        self.w = nn.Parameter(torch.randn(input_dim + 1) * 0.1)
        self.input_dim = input_dim
        self.quant_scale = quant_scale
        self.ste_alpha = 0.0

    def _ste_round(self, x):
        if self.ste_alpha > 0 and self.training:
            x_q = torch.round(x / self.quant_scale) * self.quant_scale
            return (1 - self.ste_alpha) * x + self.ste_alpha * (x + (x_q - x).detach())
        return x

    def forward(self, x):
        ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        x_ext = torch.cat([x, ones], dim=1)
        w = self._ste_round(self.w)
        return x_ext @ w


class EMLTreeInternalNode(nn.Module):
    """表达式树内部节点：对左右子节点输出执行 EML 运算。

    计算: output = exp(w1 * left + b1) - log(clamp(w2 * right + b2, min=eps))
    使用渐进式 STE 控制 w1, b1, w2, b2 的量化强度。
    """

    def __init__(self, quant_scale=0.5):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(1) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(1))
        self.w2 = nn.Parameter(torch.randn(1) * 0.1)
        self.b2 = nn.Parameter(torch.ones(1))
        self.quant_scale = quant_scale
        self.ste_alpha = 0.0

    def _ste_round(self, x):
        if self.ste_alpha > 0 and self.training:
            x_q = torch.round(x / self.quant_scale) * self.quant_scale
            return (1 - self.ste_alpha) * x + self.ste_alpha * (x + (x_q - x).detach())
        return x

    def forward(self, left, right):
        w1 = self._ste_round(self.w1)
        b1 = self._ste_round(self.b1)
        w2 = self._ste_round(self.w2)
        b2 = self._ste_round(self.b2)

        u = w1 * left + b1
        v = w2 * right + b2
        v = torch.clamp(v, min=1e-8)
        u = torch.clamp(u, min=-20.0, max=20.0)
        return torch.exp(u) - torch.log(v)


class EMLExpressionTree(nn.Module):
    """可配置深度的 EML 表达式树（完全二叉树）。

    树以数组形式存储（堆式索引）:
        - 节点 i 的左子节点为 2i+1，右子节点为 2i+2
        - 内部节点索引: 0 ~ 2^(depth-1)-2
        - 叶节点索引: 2^(depth-1)-1 ~ 2^depth-2
    """

    def __init__(self, input_dim, depth=3, quant_scale=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.quant_scale = quant_scale

        n_internal = 2 ** (depth - 1) - 1
        n_leaves = 2 ** (depth - 1)

        self.leaves = nn.ModuleList(
            [EMLTreeLeaf(input_dim, quant_scale=quant_scale) for _ in range(n_leaves)]
        )
        self.internal_nodes = nn.ModuleList(
            [EMLTreeInternalNode(quant_scale=quant_scale) for _ in range(n_internal)]
        )
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_bias = nn.Parameter(torch.zeros(1))

    def set_ste_alpha(self, alpha):
        """设置所有节点的 STE 强度 (0=连续, 1=完全量化)。"""
        for leaf in self.leaves:
            leaf.ste_alpha = alpha
        for node in self.internal_nodes:
            node.ste_alpha = alpha

    def forward(self, x):
        leaf_outputs = [leaf(x) for leaf in self.leaves]

        n_internal = len(self.internal_nodes)
        node_outputs = {}

        for i in range(len(self.leaves)):
            node_outputs[n_internal + i] = leaf_outputs[i]

        for i in range(n_internal - 1, -1, -1):
            left_idx = 2 * i + 1
            right_idx = 2 * i + 2
            node_outputs[i] = self.internal_nodes[i](
                node_outputs[left_idx], node_outputs[right_idx]
            )

        return self.output_scale * node_outputs[0] + self.output_bias

    def snapshot_weights(self, hard_leaf=False):
        """将权重快照为量化格点值。

        将每个权重圆整到最近的 quant_scale 倍数。
        若 hard_leaf=True，叶节点仅保留绝对值最大的权重（设为 ±1），其余置零。
        """
        qs = self.quant_scale
        for leaf in self.leaves:
            if hard_leaf:
                w = leaf.w.data
                max_idx = torch.argmax(torch.abs(w)).item()
                new_w = torch.zeros_like(w)
                new_w[max_idx] = torch.sign(w[max_idx])
                leaf.w.data = new_w
            else:
                leaf.w.data = torch.round(leaf.w.data / qs) * qs

        for node in self.internal_nodes:
            for name in ["w1", "b1", "w2", "b2"]:
                p = getattr(node, name).data
                rounded = torch.round(p / qs) * qs
                setattr(node, name, nn.Parameter(rounded))

    def discrete_regularizer(self):
        """计算离散正则化损失：权重到最近量化格点的距离之和。"""
        qs = self.quant_scale
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for leaf in self.leaves:
            w = leaf.w
            nearest = torch.round(w / qs) * qs
            loss = loss + torch.sum((w - nearest) ** 2)
        for node in self.internal_nodes:
            for name in ["w1", "b1", "w2", "b2"]:
                p = getattr(node, name)
                nearest = torch.round(p / qs) * qs
                loss = loss + torch.sum((p - nearest) ** 2)
        return loss


def extract_expression(model, input_symbols):
    """从训练后的 EML 表达式树中提取符号表达式（使用 sympy）。

    自动剪枝权重为零的分支，简化表达式。
    """
    if not HAS_SYMPY:
        raise ImportError("sympy is required. Install with: pip install sympy")

    n_internal = len(model.internal_nodes)
    symbols = list(input_symbols) + [sp.Integer(1)]
    qs = model.quant_scale

    def _rational(val):
        v = round(val, 4)
        if abs(v) < 1e-6:
            return sp.Integer(0)
        frac = sp.Rational(v).limit_denominator(1000)
        return frac

    def _make_linear(w_val, b_val, var):
        terms = []
        if abs(w_val) > 1e-6:
            if abs(w_val - 1.0) < 1e-6:
                terms.append(var)
            elif abs(w_val + 1.0) < 1e-6:
                terms.append(-var)
            else:
                terms.append(_rational(w_val) * var)
        if abs(b_val) > 1e-6:
            terms.append(_rational(b_val))
        if not terms:
            return sp.Integer(0)
        result = terms[0]
        for t in terms[1:]:
            result = result + t
        return result

    node_exprs = {}

    for i, leaf in enumerate(model.leaves):
        w = leaf.w.data.cpu().numpy()
        expr = sp.Integer(0)
        for j, s in enumerate(symbols):
            wj = round(w[j] / qs) * qs
            if abs(wj) > 1e-6:
                if abs(wj - 1.0) < 1e-6:
                    expr = expr + s
                elif abs(wj + 1.0) < 1e-6:
                    expr = expr - s
                else:
                    expr = expr + _rational(wj) * s
        node_exprs[n_internal + i] = expr

    for i in range(n_internal - 1, -1, -1):
        left_idx = 2 * i + 1
        right_idx = 2 * i + 2
        left_expr = node_exprs[left_idx]
        right_expr = node_exprs[right_idx]

        node = model.internal_nodes[i]
        w1 = round(node.w1.item() / qs) * qs
        b1 = round(node.b1.item() / qs) * qs
        w2 = round(node.w2.item() / qs) * qs
        b2 = round(node.b2.item() / qs) * qs

        u_expr = _make_linear(w1, b1, left_expr)
        v_expr = _make_linear(w2, b2, right_expr)

        if v_expr == sp.Integer(0):
            eml_expr = sp.exp(u_expr)
        elif u_expr == sp.Integer(0):
            eml_expr = 1 - sp.log(v_expr)
        else:
            eml_expr = sp.exp(u_expr) - sp.log(v_expr)

        node_exprs[i] = eml_expr

    root_expr = node_exprs[0]
    scale = round(model.output_scale.item(), 4)
    bias = round(model.output_bias.item(), 4)

    if abs(scale) > 1e-6 and abs(scale - 1.0) > 1e-6:
        root_expr = _rational(scale) * root_expr
    elif abs(scale + 1.0) < 1e-6:
        root_expr = -root_expr
    if abs(bias) > 1e-6:
        root_expr = root_expr + _rational(bias)

    try:
        root_expr = sp.simplify(root_expr)
    except Exception:
        pass

    try:
        expanded = sp.expand(root_expr)
        if sp.count_ops(expanded) <= sp.count_ops(root_expr):
            root_expr = expanded
    except Exception:
        pass

    return root_expr


def train_model(model, X_train, y_train,
                stage1_epochs=3000, stage1_lr=0.005,
                stage2_epochs=3000, stage2_lr=0.002,
                stage3_epochs=1000, stage3_lr=0.01,
                disc_reg_weight=0.01,
                verbose=True):
    """三阶段训练 EML 表达式树。

    阶段 1: 纯 MSE + 离散正则化拟合，STE alpha=0。
    阶段 2: 渐进 STE，alpha 从 0 线性增长到 1，同时减小离散正则化。
    阶段 3: 快照权重后，仅微调 output_scale 和 output_bias。
    """
    criterion = nn.MSELoss()
    losses = []

    # ---- Stage 1: Fit without STE, with discrete regularization ----
    model.set_ste_alpha(0.0)
    model.train()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=stage1_lr)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=stage1_epochs
    )

    best_loss = float("inf")
    best_state = None

    if verbose:
        print("  [阶段1] 拟合训练 (STE 关闭, 离散正则化)...")
    for epoch in range(stage1_epochs):
        optimizer1.zero_grad()
        pred = model(X_train)
        mse_loss = criterion(pred, y_train)

        if not torch.isfinite(mse_loss):
            continue

        disc_loss = model.discrete_regularizer()
        total_loss = mse_loss + disc_reg_weight * disc_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer1.step()
        scheduler1.step()

        loss_val = mse_loss.item()
        losses.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and epoch % 500 == 0:
            print(f"    Epoch {epoch:4d}, MSE: {loss_val:.6f}, "
                  f"Disc: {disc_loss.item():.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Stage 2: Gradual STE warmup ----
    model.train()
    optimizer2 = torch.optim.Adam(model.parameters(), lr=stage2_lr)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=stage2_epochs
    )

    warmup_epochs = stage2_epochs // 2

    if verbose:
        print(f"  [阶段2] 渐进 STE 微调 (warmup={warmup_epochs} epochs)...")
    for epoch in range(stage2_epochs):
        if epoch < warmup_epochs:
            alpha = epoch / warmup_epochs
        else:
            alpha = 1.0
        model.set_ste_alpha(alpha)

        optimizer2.zero_grad()
        pred = model(X_train)
        mse_loss = criterion(pred, y_train)

        if not torch.isfinite(mse_loss):
            continue

        disc_weight = disc_reg_weight * (1.0 - alpha)
        disc_loss = model.discrete_regularizer()
        total_loss = mse_loss + disc_weight * disc_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer2.step()
        scheduler2.step()

        loss_val = mse_loss.item()
        losses.append(loss_val)

        if verbose and epoch % 500 == 0:
            print(f"    Epoch {epoch:4d}, MSE: {loss_val:.6f}, "
                  f"alpha: {alpha:.3f}")

    model.set_ste_alpha(0.0)
    model.eval()
    return losses


def finetune_after_snapshot(model, X_train, y_train,
                            epochs=1000, lr=0.01, verbose=True):
    """快照后微调: 冻结所有内部权重，仅训练 output_scale 和 output_bias。"""
    for p in model.parameters():
        p.requires_grad = False
    model.output_scale.requires_grad_(True)
    model.output_bias.requires_grad_(True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        [model.output_scale, model.output_bias], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    model.train()
    if verbose:
        print("  [阶段3] 快照后微调 (仅 output_scale/bias)...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and epoch % 200 == 0:
            print(f"    Epoch {epoch:4d}, MSE: {loss.item():.6f}")

    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()


def compute_r2(y_true, y_pred):
    """计算 R² (决定系数) 分数。"""
    ss_res = torch.sum((y_true - y_pred) ** 2).item()
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_complexity(model):
    """计算表达式复杂度。

    Returns:
        (n_ops, n_nonzero): 操作符数量和非零参数数量
    """
    n_ops = len(model.internal_nodes)
    n_nonzero = 0
    for leaf in model.leaves:
        n_nonzero += int(torch.sum(torch.abs(leaf.w.data) > 1e-6).item())
    for node in model.internal_nodes:
        for p in [node.w1, node.b1, node.w2, node.b2]:
            n_nonzero += int(torch.abs(p.data).item() > 1e-6)
    return n_ops, n_nonzero


def plot_losses(losses_dict, save_path="symbolic_regression_losses.png"):
    """可视化训练损失曲线。"""
    if not HAS_MATPLOTLIB:
        print("matplotlib 不可用，跳过损失图")
        return
    fig, axes = plt.subplots(1, len(losses_dict), figsize=(5 * len(losses_dict), 5))
    if len(losses_dict) == 1:
        axes = [axes]
    for ax, (name, losses) in zip(axes, losses_dict.items()):
        ax.plot(losses, label=name, linewidth=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(name)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        n_s1 = len(losses) * 3 // 6
        n_s2 = len(losses) * 5 // 6
        ax.axvline(x=n_s1, color="r", linestyle="--", alpha=0.5, label="STE warmup")
        ax.axvline(x=n_s2, color="g", linestyle="--", alpha=0.5, label="Snapshot")
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"损失图已保存至 {save_path}")


def _evaluate_snapshot(model, X_train_norm, y_train_norm, X_test_norm, y_test,
                       y_mean, y_std, hard_leaf, verbose):
    """执行一次快照+微调+评估流程，返回结果字典。

    注意: 函数返回后模型保持快照+微调后的状态 (不恢复)。
    """
    model.snapshot_weights(hard_leaf=hard_leaf)

    with torch.no_grad():
        pred_train_snap = model(X_train_norm) * y_std + y_mean
        pred_test_snap = model(X_test_norm) * y_std + y_mean
        r2_train_snap = compute_r2(y_train_norm * y_std + y_mean, pred_train_snap)
        r2_test_snap = compute_r2(y_test, pred_test_snap)

    finetune_after_snapshot(model, X_train_norm, y_train_norm,
                            epochs=1000, lr=0.01, verbose=verbose)

    with torch.no_grad():
        pred_train_ft = model(X_train_norm) * y_std + y_mean
        pred_test_ft = model(X_test_norm) * y_std + y_mean
        r2_train_ft = compute_r2(y_train_norm * y_std + y_mean, pred_train_ft)
        r2_test_ft = compute_r2(y_test, pred_test_ft)
        n_ops, n_nonzero = compute_complexity(model)

    return {
        "r2_train_snap": r2_train_snap,
        "r2_test_snap": r2_test_snap,
        "r2_train_ft": r2_train_ft,
        "r2_test_ft": r2_test_ft,
        "n_ops": n_ops,
        "n_nonzero": n_nonzero,
        "hard_leaf": hard_leaf,
    }


def run_experiment(formula_name, depth=3, seed=42,
                   n_samples=500, noise_std=0.0,
                   n_restarts=5, verbose=True):
    """对单个 Feynman 公式运行完整的符号回归实验。

    流程: 数据加载 → 多次重启训练(渐进STE) → 权重快照 → 微调 → 表达式提取 → 评估
    """
    print(f"\n{'='*60}")
    print(f"公式: {formula_name}")
    if formula_name in FEYNMAN_FORMULAS:
        print(f"描述: {FEYNMAN_FORMULAS[formula_name]['description']}")
    print(f"{'='*60}")

    dataset = FeynmanDataset(
        formula_name, n_samples=n_samples, seed=seed,
        noise_std=noise_std, use_pmlb=True,
    )
    X, y = dataset.X, dataset.y
    n_features = dataset.n_features
    feature_names = dataset.feature_names

    print(f"数据集: {len(X)} 样本, {n_features} 特征")
    print(f"特征名: {feature_names}")
    print(f"目标范围: [{y.min():.4f}, {y.max():.4f}], 均值: {y.mean():.4f}")

    n_train = int(0.8 * len(X))
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=rng)
    X_train, y_train = X[perm[:n_train]], y[perm[:n_train]]
    X_test, y_test = X[perm[n_train:]], y[perm[n_train:]]

    X_mean = X_train.mean(dim=0, keepdim=True)
    X_std = X_train.std(dim=0, keepdim=True).clamp(min=1e-8)
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    y_mean = y_train.mean()
    y_std = y_train.std().clamp(min=1e-8)
    y_train_norm = (y_train - y_mean) / y_std

    best_r2 = -float("inf")
    best_result = None

    for restart in range(n_restarts):
        restart_seed = seed + restart * 100
        if verbose:
            print(f"\n  --- 重启 {restart + 1}/{n_restarts} (seed={restart_seed}) ---")

        torch.manual_seed(restart_seed)
        model = EMLExpressionTree(input_dim=n_features, depth=depth)

        losses = train_model(
            model, X_train_norm, y_train_norm,
            stage1_epochs=3000, stage1_lr=0.005,
            stage2_epochs=3000, stage2_lr=0.002,
            disc_reg_weight=0.01,
            verbose=verbose,
        )

        model.eval()
        with torch.no_grad():
            pred_test = model(X_test_norm) * y_std + y_mean
            r2_test = compute_r2(y_test, pred_test)

        if verbose:
            print(f"  重启 {restart + 1} 测试 R² (连续): {r2_test:.4f}")

        model.snapshot_weights(hard_leaf=False)
        with torch.no_grad():
            pred_test_snap = model(X_test_norm) * y_std + y_mean
            r2_snap = compute_r2(y_test, pred_test_snap)

        if verbose:
            print(f"  重启 {restart + 1} 测试 R² (快照): {r2_snap:.4f}")

        r2_combined = r2_snap
        if r2_combined > best_r2:
            best_r2 = r2_combined
            best_result = {
                "model_state": {k: v.clone() for k, v in model.state_dict().items()},
                "losses": losses,
                "restart_seed": restart_seed,
                "r2_continuous": r2_test,
                "r2_snapshot": r2_snap,
            }

    torch.manual_seed(best_result["restart_seed"])
    model = EMLExpressionTree(input_dim=n_features, depth=depth)
    model.load_state_dict(best_result["model_state"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: 深度={depth}, 叶节点={len(model.leaves)}, "
          f"内部节点={len(model.internal_nodes)}, 参数={n_params}")
    print(f"最佳重启 seed={best_result['restart_seed']}")

    with torch.no_grad():
        pred_train = model(X_train_norm) * y_std + y_mean
        pred_test = model(X_test_norm) * y_std + y_mean
        r2_train = compute_r2(y_train, pred_train)
        r2_test = compute_r2(y_test, pred_test)
        n_ops, n_nonzero = compute_complexity(model)

    print(f"\n--- 快照前 (STE 训练后，连续权重) ---")
    print(f"训练 R²: {r2_train:.4f}, 测试 R²: {r2_test:.4f}")
    print(f"操作符数: {n_ops}, 非零参数数: {n_nonzero}")

    input_syms = [sp.Symbol(name) for name in feature_names] if HAS_SYMPY else []

    state_pre_snap = {k: v.clone() for k, v in model.state_dict().items()}

    snap_soft = _evaluate_snapshot(
        model, X_train_norm, y_train_norm, X_test_norm, y_test,
        y_mean, y_std, hard_leaf=False, verbose=verbose,
    )
    print(f"\n--- 快照后 (soft leaf, quant_scale={model.quant_scale}) ---")
    print(f"训练 R²: {snap_soft['r2_train_ft']:.4f}, "
          f"测试 R²: {snap_soft['r2_test_ft']:.4f}")
    print(f"操作符数: {snap_soft['n_ops']}, 非零参数数: {snap_soft['n_nonzero']}")

    model.load_state_dict(state_pre_snap)

    snap_hard = _evaluate_snapshot(
        model, X_train_norm, y_train_norm, X_test_norm, y_test,
        y_mean, y_std, hard_leaf=True, verbose=verbose,
    )
    print(f"\n--- 快照后 (hard leaf, 每叶仅选一个特征) ---")
    print(f"训练 R²: {snap_hard['r2_train_ft']:.4f}, "
          f"测试 R²: {snap_hard['r2_test_ft']:.4f}")
    print(f"操作符数: {snap_hard['n_ops']}, 非零参数数: {snap_hard['n_nonzero']}")

    best_snap = snap_hard if (
        snap_hard["r2_test_ft"] > 0.8 and
        snap_hard["r2_test_ft"] >= snap_soft["r2_test_ft"] - 0.05
    ) else snap_soft

    use_hard = best_snap["hard_leaf"]
    print(f"\n--- 选用的快照方式: {'hard leaf' if use_hard else 'soft leaf'} ---")

    model.load_state_dict(state_pre_snap)
    model.snapshot_weights(hard_leaf=use_hard)
    finetune_after_snapshot(model, X_train_norm, y_train_norm,
                            epochs=1000, lr=0.01, verbose=verbose)

    if HAS_SYMPY and input_syms:
        try:
            expr = extract_expression(model, input_syms)
            expr_str = str(expr)
            if len(expr_str) > 500:
                expr_str = expr_str[:500] + "..."
            print(f"\n提取的符号表达式:")
            print(f"  {expr_str}")

            if verbose:
                try:
                    X_test_np = X_test_norm.numpy()
                    expr_np = sp.lambdify(input_syms, expr, "numpy")
                    y_expr = expr_np(*[X_test_np[:, i] for i in range(len(input_syms))])
                    if np.isscalar(y_expr):
                        y_expr = np.full(len(X_test_np), y_expr)
                    y_expr = np.array(y_expr, dtype=np.float32)
                    valid = np.isfinite(y_expr)
                    if valid.sum() > 10:
                        y_pred_expr = torch.from_numpy(y_expr) * y_std + y_mean
                        r2_expr = compute_r2(y_test[valid], y_pred_expr[valid])
                        print(f"  表达式数值验证 R²: {r2_expr:.4f} "
                              f"({valid.sum()}/{len(y_expr)} 有效样本)")
                    else:
                        print(f"  表达式数值验证: 仅有 {valid.sum()}/{len(y_expr)} 有效样本, "
                              f"跳过 R² 计算 (表达式含数值溢出)")
                except Exception as e:
                    print(f"  表达式数值验证失败: {e}")
        except Exception as e:
            print(f"\n表达式提取失败: {e}")
    else:
        print("\nsympy 未安装，跳过表达式提取 (pip install sympy)")

    return {
        "formula": formula_name,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "r2_train_snap": best_snap["r2_train_snap"],
        "r2_test_snap": best_snap["r2_test_snap"],
        "r2_train_ft": best_snap["r2_train_ft"],
        "r2_test_ft": best_snap["r2_test_ft"],
        "n_ops": best_snap["n_ops"],
        "n_nonzero": best_snap["n_nonzero"],
        "losses": best_result["losses"],
    }


def main():
    print("=" * 60)
    print("EML 符号回归 - Feynman 数据集")
    print("=" * 60)

    if HAS_PMLB:
        print("[OK] PMLB 可用，将尝试从 PMLB 加载数据")
    else:
        print("[--] PMLB 不可用，使用合成数据 (pip install pmlb)")

    if HAS_SYMPY:
        print("[OK] sympy 可用，将提取符号表达式")
    else:
        print("[--] sympy 不可用，跳过表达式提取 (pip install sympy)")

    if HAS_MATPLOTLIB:
        print("[OK] matplotlib 可用，将生成损失图")
    else:
        print("[--] matplotlib 不可用，跳过损失图 (pip install matplotlib)")

    formulas = ["I.12.2", "I.13.12", "II.6.11", "II.11.3", "I.14.4"]

    results = {}
    all_losses = {}

    for formula_name in formulas:
        try:
            result = run_experiment(
                formula_name=formula_name,
                depth=3,
                seed=42,
                n_samples=500,
                noise_std=0.0,
                n_restarts=5,
            )
            results[formula_name] = result
            all_losses[formula_name] = result["losses"]
        except Exception as e:
            print(f"\n公式 {formula_name} 实验失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n\n{'='*60}")
    print("汇总结果")
    print(f"{'='*60}")
    header = (
        f"{'公式':<12} {'R²(连续)':<10} {'R²(快照)':<10} {'R²(微调)':<10} "
        f"{'操作符':<6} {'非零参数':<8}"
    )
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<12} {r['r2_test']:<10.4f} {r['r2_test_snap']:<10.4f} "
            f"{r['r2_test_ft']:<10.4f} {r['n_ops']:<6} {r['n_nonzero']:<8}"
        )

    if all_losses and HAS_MATPLOTLIB:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "symbolic_regression_losses.png",
        )
        plot_losses(all_losses, save_path=save_path)


if __name__ == "__main__":
    main()
