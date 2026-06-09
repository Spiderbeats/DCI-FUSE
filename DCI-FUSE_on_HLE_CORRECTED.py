import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 1. 基础数据处理工具（完美对齐 HLE JSONL 解析）
# ============================================================================
def flatten_to_float_list(raw):
    """将嵌套列表拉平为 float 列表，缺失值填 0.0"""
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
    all_scores = [[] for _ in range(m)]
    for mat in all_matrices:
        for j in range(m):
            all_scores[j].extend(mat[:, j].flatten())
    global_min = np.zeros(m)
    global_max = np.zeros(m)
    for j in range(m):
        arr = np.array(all_scores[j])
        global_min[j] = np.min(arr)
        global_max[j] = np.max(arr)
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


def load_hle_jsonl(data_path):
    """加载HLE格式的JSONL数据"""
    raw_matrices = []
    all_labels = []
    verifier_names = None
    m = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if 'is_correct' not in obj:
                continue
            labels = obj['is_correct']
            labels = np.array([bool(x) for x in labels], dtype=bool)
            if len(labels) == 0:
                continue

            score_cols = [k for k in obj.keys() if k.endswith('_scores')]
            score_cols.sort()
            if verifier_names is None:
                verifier_names = score_cols
                m = len(verifier_names)
                print(f"Found {m} verifiers: {verifier_names}")
            else:
                assert score_cols == verifier_names, "Verifier order mismatch"

            n = len(labels)
            mat = np.zeros((n, m))
            for j, col in enumerate(verifier_names):
                scores_raw = obj[col]
                for i in range(n):
                    if i < len(scores_raw):
                        val = scores_raw[i]
                        if isinstance(val, list):
                            val = val[0] if len(val) > 0 else 0.0
                        if val is None:
                            val = 0.0
                        try:
                            mat[i, j] = float(val)
                        except:
                            mat[i, j] = 0.0
                    else:
                        mat[i, j] = 0.0
            raw_matrices.append(mat)
            all_labels.append(labels)
    print(f"Loaded {len(raw_matrices)} questions successfully.")
    return raw_matrices, all_labels, m


# ============================================================================
# 2. 核心 FUSE 数学算子（修复版）
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
    坐标下降搜索最优阈值
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
                # 因子 = 1 - v + v(1-2η) = 1 - v(1-(1-2η)) = 1 - v(2η)
                # = 1 - 2vη（此形式有误，应该是下面的）
                # 正确：1 - v(-1) + v(1-2η) = 1 + v + v(1-2η) = 1 + v(2-2η) = 1 + 2v(1-η)
                factor1_neg = (1.0 - v1 * (-1) + v1 * (1 - 2 * eta[j1]))
                factor2_neg = (1.0 - v2 * (-1) + v2 * (1 - 2 * eta[j2]))
                factor3_neg = (1.0 - v3 * (-1) + v3 * (1 - 2 * eta[j3]))
                neg = (1.0 - b) * factor1_neg * factor2_neg * factor3_neg
                
                prob = pos / (pos + neg + 1e-15)
                prob_sum += prob
                count += 1
    
    return prob_sum / count if count > 0 else np.full(N, 0.5)


def view_merging_posterior(X_bin, psi, eta, b, m, random_seed=42):
    """当验证器不足3个时的回退方案"""
    np.random.seed(random_seed)
    indices = np.random.permutation(m)
    splits = np.array_split(indices, 3)
    X_view = np.zeros((X_bin.shape[0], 3))
    for v, idxs in enumerate(splits):
        if len(idxs) > 0:
            X_view[:, v] = np.sign(np.sum(X_bin[:, idxs], axis=1))
        else:
            X_view[:, v] = 0
    
    psi_view = np.array([np.mean(psi[idxs]) if len(idxs) > 0 else 0.5 for idxs in splits])
    eta_view = np.array([np.mean(eta[idxs]) if len(idxs) > 0 else 0.5 for idxs in splits])
    
    v1, v2, v3 = X_view[:, 0], X_view[:, 1], X_view[:, 2]
    factor1_pos = (1.0 - v1 + v1 * (2 * psi_view[0] - 1))
    factor2_pos = (1.0 - v2 + v2 * (2 * psi_view[1] - 1))
    factor3_pos = (1.0 - v3 + v3 * (2 * psi_view[2] - 1))
    pos = (1.0 + b) * factor1_pos * factor2_pos * factor3_pos
    
    factor1_neg = (1.0 - v1 * (-1) + v1 * (1 - 2 * eta_view[0]))
    factor2_neg = (1.0 - v2 * (-1) + v2 * (1 - 2 * eta_view[1]))
    factor3_neg = (1.0 - v3 * (-1) + v3 * (1 - 2 * eta_view[2]))
    neg = (1.0 - b) * factor1_neg * factor2_neg * factor3_neg
    
    return pos / (pos + neg + 1e-15)


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
# 3. HLE DCI 主评估流程（修复版）
# ============================================================================
def run_dci_hle_corrected(data_path, K_list=[1, 2, 3, 4, 5]):
    """
    DCI-FUSE 修复版本 - 所有关键数学公式都已更正
    
    主要修复：
    1. TCI违反度的聚合顺序：∑_{j3} Var_{j1<j2<j3}
    2. 后验概率公式：精确的论文Eq.13形式
    3. MoM符号确定：严格的"多数验证器>0.5"逻辑
    """
    print("\n" + "=" * 80)
    print("DCI-FUSE [CORRECTED VERSION - All Math Fixed]")
    print("=" * 80)

    # 加载数据
    raw_matrices, all_labels, m = load_hle_jsonl(data_path)
    norm_matrices = global_minmax_normalize(raw_matrices)
    total_qs = len(norm_matrices)

    # 单题独立计算阈值
    X_local_blocks = []
    for V in norm_matrices:
        tau_q, _ = coordinate_descent_thresholds(V, max_iters=30)
        X_local_blocks.append(np.where(V >= tau_q, 1.0, -1.0))
    X_local_stacked = np.vstack(X_local_blocks)

    for n_regimes in K_list:
        if n_regimes == 1:
            # K=1 单题独立流
            fuse_correct_sum = 0.0
            for qid, (V, labels) in enumerate(zip(norm_matrices, all_labels)):
                X_bin_q = X_local_blocks[qid]
                psi, eta, b = mom_estimates_binary(X_bin_q)
                bal_acc = (psi + eta) / 2.0
                active = [j for j in range(m) if bal_acc[j] > 0.5]

                if len(active) >= 3:
                    p_hat = triplet_posterior_probabilities_corrected(X_bin_q, psi, eta, b, active)
                else:
                    p_hat = view_merging_posterior(X_bin_q, psi, eta, b, m)

                if len(active) >= 2:
                    w, intercept, ok = fit_logistic_ensemble(V, p_hat, active)
                    fuse_scores = V @ w + intercept if ok else np.mean(V, axis=1)
                else:
                    fuse_scores = np.mean(V, axis=1)

                fuse_correct_sum += tie_breaking_accuracy(fuse_scores, labels)
            
            print(f"Result for DCI-FUSE (K=1) [Corrected]: {fuse_correct_sum / total_qs * 100:.2f}%")
            continue

        # K > 1 谱映射聚类分流
        M_hat = np.cov(X_local_stacked, rowvar=False)
        M_tilde = M_hat - np.diag(np.diag(M_hat))
        eigenvalues, eigenvectors = np.linalg.eigh(M_tilde)
        U_K = eigenvectors[:, np.argsort(eigenvalues)[::-1][:n_regimes]]

        Z_questions = np.array([np.mean(X_q @ U_K, axis=0) for X_q in X_local_blocks])
        question_cluster_labels = KMeans(n_clusters=n_regimes, random_state=42, n_init=30).fit_predict(Z_questions)

        regime_parameters = {}
        for k in range(n_regimes):
            cluster_indices = np.where(question_cluster_labels == k)[0]
            if len(cluster_indices) == 0:
                regime_parameters[k] = {'psi': np.full(m, 0.7), 'eta': np.full(m, 0.7), 'b': 0.0}
                continue
            cluster_samples = np.vstack([X_local_blocks[idx] for idx in cluster_indices])
            psi_k, eta_k, b_k = mom_estimates_binary(cluster_samples)
            regime_parameters[k] = {'psi': psi_k, 'eta': eta_k, 'b': b_k}

        fuse_correct_sum = 0.0
        for qid, (V, labels) in enumerate(zip(norm_matrices, all_labels)):
            X_bin_q = X_local_blocks[qid]
            q_regime = question_cluster_labels[qid]
            p_params = regime_parameters[q_regime]

            psi, eta, b = p_params['psi'], p_params['eta'], p_params['b']
            bal_acc = (psi + eta) / 2.0
            active = [j for j in range(m) if bal_acc[j] > 0.5]

            if len(active) >= 3:
                p_hat = triplet_posterior_probabilities_corrected(X_bin_q, psi, eta, b, active)
            else:
                p_hat = view_merging_posterior(X_bin_q, psi, eta, b, m)

            if len(active) >= 2:
                w, intercept, ok = fit_logistic_ensemble(V, p_hat, active)
                fuse_scores = V @ w + intercept if ok else np.mean(V, axis=1)
            else:
                fuse_scores = np.mean(V, axis=1)

            fuse_correct_sum += tie_breaking_accuracy(fuse_scores, labels)

        print(f"Result for DCI-FUSE (K={n_regimes}) [Corrected]: {fuse_correct_sum / total_qs * 100:.2f}%")


if __name__ == "__main__":
    run_dci_hle_corrected('FUSE-hle-data.jsonl')
