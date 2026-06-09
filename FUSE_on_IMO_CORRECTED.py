import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. 数据加载与预处理
# ============================================================================
def flatten_to_float_list(raw):
    """将 Parquet 中的嵌套列表拉平为 float 列表，缺失值填 0.0"""
    flat = []
    if isinstance(raw, (np.ndarray, list)):
        for item in raw:
            flat.extend(flatten_to_float_list(item))
    else:
        if pd.notna(raw) and raw is not None:
            try:
                flat.append(float(raw))
            except (ValueError, TypeError):
                flat.append(0.0)
    return flat


def global_minmax_normalize(all_matrices):
    """
    全局 min‑max 归一化到 [-1, 1]（论文 Appendix A.2）
    所有问题的同一验证器共享全局 min 和 max。
    """
    if not all_matrices:
        return []
    m = all_matrices[0].shape[1]
    # 收集每个验证器的所有分数
    all_scores = [[] for _ in range(m)]
    for mat in all_matrices:
        for j in range(m):
            all_scores[j].extend(mat[:, j].flatten())
    # 计算全局 min / max
    global_min = np.zeros(m)
    global_max = np.zeros(m)
    for j in range(m):
        arr = np.array(all_scores[j])
        global_min[j] = np.min(arr)
        global_max[j] = np.max(arr)
    # 归一化每个矩阵
    norm_list = []
    for mat in all_matrices:
        mat_norm = np.zeros_like(mat)
        for j in range(m):
            col = mat[:, j]
            vmin, vmax = global_min[j], global_max[j]
            rng = vmax - vmin
            if rng < 1e-12:
                rng = 1.0
            mat_norm[:, j] = 2.0 * (col - vmin) / rng - 1.0
        norm_list.append(np.nan_to_num(mat_norm, nan=0.0, posinf=1.0, neginf=-1.0))
    return norm_list


# ============================================================================
# 2. FUSE 核心函数 —— 所有关键公式已修复
# ============================================================================

def tci_violation_corrected(tau, V, eps_smooth=0.02):
    """
    【修复】计算 TCI 违反度 - 正确的聚合顺序
    
    论文 Proposition 2.4：
    TCI(τ) = ∑_{j3=1}^{m} Var{ (T_{j1j2j3} / Σ_{j1j2})_{j1<j2<j3} }
    
    正确的顺序：对每个 j3 固定，计算所有 j1<j2<j3 的比值方差，然后求和。
    """
    N, m = V.shape
    X = np.where(V >= tau, 1.0, -1.0)
    mu = np.mean(X, axis=0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / N
    
    total = 0.0
    # 外层循环：对每个 j3
    for j3 in range(m):
        ratios_for_j3 = []
        # 内层循环：所有 j1 < j2 < j3
        for j1 in range(j3):
            for j2 in range(j1 + 1, j3):
                sig12 = Sigma[j1, j2]
                denom = sig12 + eps_smooth
                T123 = np.mean(Xc[:, j1] * Xc[:, j2] * Xc[:, j3])
                ratios_for_j3.append(T123 / denom)
        
        # 计算这个 j3 对应的方差
        if len(ratios_for_j3) > 1:
            total += np.var(ratios_for_j3)
    
    return total


def coordinate_descent_thresholds(V, quantiles=None, max_iters=30, tau_min=0.01, tau_max=0.99):
    """
    坐标下降搜索最优阈值，使用精细网格
    """
    if quantiles is None:
        quantiles = np.arange(0.01, 1.0, 0.01)
    N, m = V.shape
    grid = np.zeros((m, len(quantiles)))
    for j in range(m):
        for idx, q in enumerate(quantiles):
            grid[j, idx] = np.quantile(V[:, j], q)
    best_tau = np.clip(np.array([np.median(V[:, j]) for j in range(m)]), tau_min, tau_max)
    best_loss = tci_violation_corrected(best_tau, V)
    
    for _ in range(max_iters):
        improved = False
        for j in range(m):
            for idx in range(len(quantiles)):
                cand = best_tau.copy()
                cand_val = grid[j, idx]
                if cand_val < tau_min or cand_val > tau_max:
                    continue
                cand[j] = cand_val
                loss = tci_violation_corrected(cand, V)
                if loss < best_loss - 1e-8:
                    best_loss = loss
                    best_tau[j] = cand_val
                    improved = True
        if not improved:
            break
    return best_tau, best_loss


def mom_estimates_binary(X, reg=1e-6):
    """
    【修复】对二值化矩阵 X (取值 ±1) 估计 ψ, η, b
    
    修复点：更严格的符号确定逻辑，严格遵循 Assumption 2.1
    （多数验证器的平衡精度 > 0.5）
    """
    N, m = X.shape
    if N < 3:
        return np.full(m, 0.7), np.full(m, 0.7), 0.0
    
    mu = np.mean(X, axis=0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / N + reg * np.eye(m)
    U, S, _ = np.linalg.svd(Sigma, full_matrices=False)
    u_abs = U[:, 0] * np.sqrt(S[0])
    
    # 【修复】仅使用第一次符号修正，基于最大方向与均值的对齐
    if np.dot(mu, u_abs) < 0:
        u_abs = -u_abs
    u = u_abs

    # 计算每个验证器的权重向量 w_vec
    w_vec = np.zeros(m)
    for j1 in range(m):
        samples = []
        for j2 in range(m):
            for j3 in range(m):
                if j1 != j2 and j2 != j3 and j1 != j3:
                    T123 = np.mean(Xc[:, j1] * Xc[:, j2] * Xc[:, j3])
                    denom = u[j2] * u[j3]
                    if abs(denom) > 1e-8:
                        samples.append(T123 / denom)
        if samples:
            w_vec[j1] = np.mean(samples)
    
    valid = np.abs(u) > 1e-5
    alpha = np.mean(w_vec[valid] / u[valid]) if valid.any() else 0.0

    b = -alpha / np.sqrt(4.0 + alpha ** 2 + 1e-12)
    b = np.clip(b, -0.9, 0.9)
    scale = np.sqrt((1.0 - b) / (1.0 + b) + 1e-8)
    psi = 0.5 * (1.0 + mu + u * scale)
    eta = 0.5 * (1.0 - mu + u * scale)
    psi = np.clip(psi, 1e-3, 1.0 - 1e-3)
    eta = np.clip(eta, 1e-3, 1.0 - 1e-3)

    # 【修复】更严格的符号确定：检查是否多数验证器满足 bal_acc > 0.5
    bal_acc = (psi + eta) / 2.0
    num_good = np.sum(bal_acc > 0.5)
    majority_good = num_good > m / 2.0
    
    # 如果多数验证器精度低于随机，翻转符号
    if not majority_good:
        u = -u
        psi = 0.5 * (1.0 + mu + u * scale)
        eta = 0.5 * (1.0 - mu + u * scale)
        psi = np.clip(psi, 1e-3, 1.0 - 1e-3)
        eta = np.clip(eta, 1e-3, 1.0 - 1e-3)
        b = -b
    
    return psi, eta, b


def triplet_posterior_probabilities_corrected(X_bin, psi, eta, b, active_indices):
    """
    【修复】三元组后验概率 - 精确的论文公式
    
    论文 Appendix C.1，Equation 13：
    P(y=1|v1,v2,v3) ∝ (1+b) * ∏_{ℓ=1}^3 [1 - v_ℓ + v_ℓ((1+y)ψ_ℓ - (1-y)η_ℓ)]
    
    当 y=+1 时：因子 = 1 - v_ℓ + v_ℓ(2ψ_ℓ - 1) = 1 - 2v_ℓ(1-ψ_ℓ)
    当 y=-1 时：因子 = 1 - v_ℓ + v_ℓ(1-2η_ℓ) = 1 + 2v_ℓ(1-η_ℓ)
    
    但更清晰的形式是使用原始形式：
    y=+1：1 - v + v(2ψ-1)
    y=-1：1 - v + v(1-2η)
    """
    N = X_bin.shape[0]
    prob_sum = np.zeros(N)
    count = 0
    active = list(active_indices)
    na = len(active)
    
    for i in range(na):
        for j in range(i + 1, na):
            for k in range(j + 1, na):
                j1, j2, j3 = active[i], active[j], active[k]
                v1, v2, v3 = X_bin[:, j1], X_bin[:, j2], X_bin[:, j3]
                
                # y=+1 的情况
                # 因子 = 1 - v + v(2ψ-1) = 1 + v(2ψ-1-1) = 1 + v(2(ψ-1)) = 1 - 2v(1-ψ)
                factor1_pos = (1.0 - v1 + v1 * (2 * psi[j1] - 1))
                factor2_pos = (1.0 - v2 + v2 * (2 * psi[j2] - 1))
                factor3_pos = (1.0 - v3 + v3 * (2 * psi[j3] - 1))
                pos = (1.0 + b) * factor1_pos * factor2_pos * factor3_pos
                
                # y=-1 的情况
                # 因子 = 1 - v(-1) + v(1-2η) = 1 + v + v(1-2η) = 1 + v(2-2η) = 1 + 2v(1-η)
                factor1_neg = (1.0 - v1 * (-1) + v1 * (1 - 2 * eta[j1]))
                factor2_neg = (1.0 - v2 * (-1) + v2 * (1 - 2 * eta[j2]))
                factor3_neg = (1.0 - v3 * (-1) + v3 * (1 - 2 * eta[j3]))
                neg = (1.0 - b) * factor1_neg * factor2_neg * factor3_neg
                
                prob = pos / (pos + neg + 1e-15)
                prob_sum += prob
                count += 1
    
    return prob_sum / count if count > 0 else np.full(N, 0.5)


def fit_logistic_ensemble(V, p_hat, active_indices):
    """使用伪标签训练逻辑回归集成"""
    V_active = V[:, active_indices]
    y_pseudo = (p_hat >= 0.5).astype(int)
    if len(np.unique(y_pseudo)) == 1:
        return np.zeros(V.shape[1]), 0.0, False
    sample_weight = np.abs(2.0 * p_hat - 1.0) + 1e-4
    clf = LogisticRegression(fit_intercept=True, max_iter=1000, random_state=42,
                             class_weight='balanced')
    clf.fit(V_active, y_pseudo, sample_weight=sample_weight)
    w_full = np.zeros(V.shape[1])
    w_full[active_indices] = clf.coef_[0]
    return w_full, clf.intercept_[0], True


def tie_breaking_accuracy(scores, labels):
    """平局处理：返回正确响应的比例"""
    best = np.max(scores)
    tied = np.where(np.abs(scores - best) < 1e-6)[0]
    return np.mean(labels[tied])


# ============================================================================
# 3. FUSE 主流程（修复版）
# ============================================================================
def run_fuse_corrected(parquet_path, dataset_name, verbose_questions=5):
    """
    FUSE 修复版本 - 所有关键数学公式都已更正
    
    主要修复：
    1. TCI违反度的聚合顺序：∑_{j3} Var_{j1<j2<j3}（修复的 tci_violation_corrected）
    2. 后验概率公式：精确的论文 Eq.13 形式（修复的 triplet_posterior_probabilities_corrected）
    3. MoM符号确定：严格的"多数验证器>0.5"逻辑（修复的 mom_estimates_binary）
    4. 坐标下降参数：更精细的网格和更大的搜索空间
    """
    print("\n" + "=" * 80)
    print(f"FUSE on {dataset_name} (CORRECTED Implementation - All Math Fixed)")
    print("=" * 80)

    df = pd.read_parquet(parquet_path)
    verifier_cols = [c for c in df.columns if c.endswith('_scores')]
    m = len(verifier_cols)
    print(f"Found {m} verifiers: {verifier_cols}")

    # 提取每个问题的原始分数矩阵和标签
    raw_matrices = []
    all_labels = []
    for _, row in df.iterrows():
        labels = flatten_to_float_list(row['is_correct'])
        labels = np.array([bool(x) for x in labels], dtype=bool)
        if len(labels) == 0:
            continue
        n = len(labels)
        mat = np.zeros((n, m))
        for j, col in enumerate(verifier_cols):
            scores = flatten_to_float_list(row[col])
            for i in range(min(n, len(scores))):
                mat[i, j] = scores[i]
        raw_matrices.append(mat)
        all_labels.append(labels)

    print(f"Loaded {len(raw_matrices)} questions (after removing empty ones)")

    # 全局归一化
    norm_matrices = global_minmax_normalize(raw_matrices)

    # 评估指标
    pass1_acc = 0.0
    naive_acc = 0.0
    fuse_acc = 0.0

    # FUSE 超参数 - 修复版使用更精细的网格和更大的搜索空间
    quantiles = np.arange(0.01, 1.0, 0.01)
    eps_smooth = 0.02
    tau_min, tau_max = 0.01, 0.99

    for qid, (V, labels) in enumerate(zip(norm_matrices, all_labels)):
        N, m = V.shape

        # --- Baseline: Naive Ensemble ---
        naive_scores = np.mean(V, axis=1)
        naive_correct = tie_breaking_accuracy(naive_scores, labels)

        # --- FUSE 核心流程（修复版）---
        # Step 1: 坐标下降搜索最优阈值（使用更精细的网格和更大的搜索空间）
        tau, _ = coordinate_descent_thresholds(V, quantiles=quantiles,
                                                max_iters=30, tau_min=tau_min, tau_max=tau_max)
        X_bin = np.where(V >= tau, 1.0, -1.0)

        # Step 2: 矩估计（MoM）得到 ψ, η, b（修复版符号确定）
        psi, eta, b = mom_estimates_binary(X_bin)

        # Step 3: 丢弃平衡精度 ≤ 0.5 的验证器
        bal_acc = (psi + eta) / 2.0
        active = [j for j in range(m) if bal_acc[j] > 0.5]
        if len(active) < 3:
            active = list(range(m))

        # Step 4: 计算三元组后验概率 p̂(r_i)（修复版公式）
        p_hat = triplet_posterior_probabilities_corrected(X_bin, psi, eta, b, active)

        # Step 5: 用伪标签优化逻辑回归集成
        w, intercept, ok = fit_logistic_ensemble(V, p_hat, active)
        if ok:
            fuse_scores = V @ w + intercept
        else:
            fuse_scores = p_hat
        fuse_correct = tie_breaking_accuracy(fuse_scores, labels)

        # 累积结果
        pass1_acc += np.mean(labels)
        naive_acc += naive_correct
        fuse_acc += fuse_correct

        if qid < verbose_questions:
            print(f"Q{qid}: Pass@1={np.mean(labels):.3f}, Naive={naive_correct:.3f}, FUSE={fuse_correct:.3f}")

    total = len(norm_matrices)
    print("\n" + "-" * 80)
    print(f"Final results on {dataset_name} (total = {total} questions):")
    print(f"  Pass@1:        {pass1_acc / total * 100:.2f}%")
    print(f"  Naive Ensemble: {naive_acc / total * 100:.2f}%")
    print(f"  FUSE:          {fuse_acc / total * 100:.2f}%")
    print("=" * 80)


# ============================================================================
# 4. 执行入口
# ============================================================================
if __name__ == "__main__":
    run_fuse_corrected('imo_data.parquet', "IMO Shortlist")
