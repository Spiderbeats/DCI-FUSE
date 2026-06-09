import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 1. 基础数据处理工具（原始版本保留）
# ============================================================================
def flatten_to_float_list(raw):
    flat = []
    if isinstance(raw, (np.ndarray, list)):
        for item in raw: flat.extend(flatten_to_float_list(item))
    else:
        if pd.notna(raw) and raw is not None:
            try:
                flat.append(float(raw))
            except (ValueError, TypeError):
                flat.append(0.0)
    return flat


def global_minmax_normalize(all_matrices):
    if not all_matrices: return []
    m = all_matrices[0].shape[1]
    all_scores = [[] for _ in range(m)]
    for mat in all_matrices:
        for j in range(m): all_scores[j].extend(mat[:, j].flatten())
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
            if rng < 1e-12: rng = 1.0
            mat_norm[:, j] = 2.0 * (col - vmin) / rng - 1.0
        norm_list.append(np.nan_to_num(mat_norm, nan=0.0, posinf=1.0, neginf=-1.0))
    return norm_list


def load_hle_jsonl(data_path):
    raw_matrices = []
    all_labels = []
    verifier_names = None
    m = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if 'is_correct' not in obj: continue
            labels = obj['is_correct']
            labels = np.array([bool(x) for x in labels], dtype=bool)
            if len(labels) == 0: continue

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
                        if isinstance(val, list): val = val[0] if len(val) > 0 else 0.0
                        if val is None: val = 0.0
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
# 2. 核心 FUSE 数学算子（仅修改后验概率，其他保持原始）
# ============================================================================

def tci_violation(tau, V, eps_smooth=0.02):
    """【保留原始】计算 TCI 违反度"""
    N, m = V.shape
    X = np.where(V >= tau, 1.0, -1.0)
    mu = np.mean(X, axis=0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / N
    total = 0.0
    for j1 in range(m):
        for j2 in range(j1 + 1, m):
            sig12 = Sigma[j1, j2]
            denom = sig12 + eps_smooth
            ratios = []
            for j3 in range(m):
                if j3 == j1 or j3 == j2: continue
                T123 = np.mean(Xc[:, j1] * Xc[:, j2] * Xc[:, j3])
                ratios.append(T123 / denom)
            if len(ratios) > 1: total += np.var(ratios)
    return total


def coordinate_descent_thresholds(V, quantiles=None, max_iters=30, tau_min=0.01, tau_max=0.99):
    """【保留原始】坐标下降搜索最优阈值"""
    if quantiles is None: quantiles = np.arange(0.01, 1.0, 0.01)
    N, m = V.shape
    grid = np.zeros((m, len(quantiles)))
    for j in range(m):
        for idx, q in enumerate(quantiles): grid[j, idx] = np.quantile(V[:, j], q)
    best_tau = np.clip(np.array([np.median(V[:, j]) for j in range(m)]), tau_min, tau_max)
    best_loss = tci_violation(best_tau, V)
    for _ in range(max_iters):
        improved = False
        for j in range(m):
            for idx in range(len(quantiles)):
                cand = best_tau.copy()
                cand_val = grid[j, idx]
                if cand_val < tau_min or cand_val > tau_max: continue
                cand[j] = cand_val
                loss = tci_violation(cand, V)
                if loss < best_loss - 1e-8:
                    best_loss = loss
                    best_tau[j] = cand_val
                    improved = True
        if not improved: break
    return best_tau, best_loss


def mom_estimates_binary(X, reg=1e-6):
    """【保留原始】MoM 估计"""
    N, m = X.shape
    if N < 3: return np.full(m, 0.7), np.full(m, 0.7), 0.0
    mu = np.mean(X, axis=0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / N + reg * np.eye(m)
    U, S, _ = np.linalg.svd(Sigma, full_matrices=False)
    u_abs = U[:, 0] * np.sqrt(S[0])
    if np.dot(mu, u_abs) < 0: u_abs = -u_abs
    u = u_abs

    w_vec = np.zeros(m)
    for j1 in range(m):
        samples = []
        for j2 in range(m):
            for j3 in range(m):
                if j1 != j2 and j2 != j3 and j1 != j3:
                    T123 = np.mean(Xc[:, j1] * Xc[:, j2] * Xc[:, j3])
                    denom = u[j2] * u[j3]
                    if abs(denom) > 1e-8: samples.append(T123 / denom)
        if samples: w_vec[j1] = np.mean(samples)
    valid = np.abs(u) > 1e-5
    alpha = np.mean(w_vec[valid] / u[valid]) if valid.any() else 0.0

    b = -alpha / np.sqrt(4.0 + alpha ** 2 + 1e-12)
    b = np.clip(b, -0.9, 0.9)
    scale = np.sqrt((1.0 - b) / (1.0 + b) + 1e-8)
    psi = 0.5 * (1.0 + mu + u * scale)
    eta = 0.5 * (1.0 - mu + u * scale)
    psi = np.clip(psi, 1e-3, 1.0 - 1e-3)
    eta = np.clip(eta, 1e-3, 1.0 - 1e-3)

    if np.mean((psi + eta) / 2.0) < 0.5:
        u = -u
        psi = 0.5 * (1.0 + mu + u * scale)
        eta = 0.5 * (1.0 - mu + u * scale)
        psi = np.clip(psi, 1e-3, 1.0 - 1e-3)
        eta = np.clip(eta, 1e-3, 1.0 - 1e-3)
        b = -b
    return psi, eta, b


def triplet_posterior_probabilities_original(X_bin, psi, eta, b, active_indices):
    """【原始版本】后验概率 - 保留您的原始实现"""
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
                pos = (1.0 + b) * ((1.0 + v1 * (2 * psi[j1] - 1)) * (1.0 + v2 * (2 * psi[j2] - 1)) * (
                            1.0 + v3 * (2 * psi[j3] - 1)))
                neg = (1.0 - b) * ((1.0 + v1 * (1 - 2 * eta[j1])) * (1.0 + v2 * (1 - 2 * eta[j2])) * (
                            1.0 + v3 * (1 - 2 * eta[j3])))
                prob = pos / (pos + neg + 1e-15)
                prob_sum += prob
                count += 1
    return prob_sum / count if count > 0 else np.full(N, 0.5)


def triplet_posterior_probabilities_fixed(X_bin, psi, eta, b, active_indices):
    """
    【修复版本】后验概率 - 仅修改因子计算，保留其他逻辑
    
    原始公式问题：使用了 pos = (1+b) * ((1 + v(2ψ-1)) * ...)
                    使用了 neg = (1-b) * ((1 + v(1-2η)) * ...)
    
    修复公式：使用精确的因子形式
                    pos因子 = 1 - v + v(2ψ-1)  （y=+1）
                    neg因子 = 1 - v(-1) + v(1-2η)  （y=-1）
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
                
                # 【修复】使用精确的因子形式（y=+1）
                factor1_pos = (1.0 - v1 + v1 * (2 * psi[j1] - 1))
                factor2_pos = (1.0 - v2 + v2 * (2 * psi[j2] - 1))
                factor3_pos = (1.0 - v3 + v3 * (2 * psi[j3] - 1))
                pos = (1.0 + b) * factor1_pos * factor2_pos * factor3_pos
                
                # 【修复】使用精确的因子形式（y=-1）
                factor1_neg = (1.0 - v1 * (-1) + v1 * (1 - 2 * eta[j1]))
                factor2_neg = (1.0 - v2 * (-1) + v2 * (1 - 2 * eta[j2]))
                factor3_neg = (1.0 - v3 * (-1) + v3 * (1 - 2 * eta[j3]))
                neg = (1.0 - b) * factor1_neg * factor2_neg * factor3_neg
                
                prob = pos / (pos + neg + 1e-15)
                prob_sum += prob
                count += 1
    return prob_sum / count if count > 0 else np.full(N, 0.5)


def view_merging_posterior(X_bin, psi, eta, b, m, random_seed=42):
    """【保留原始】当验证器不足3个时的回退方案"""
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
    pos = (1.0 + b) * ((1.0 + v1 * (2 * psi_view[0] - 1)) * (1.0 + v2 * (2 * psi_view[1] - 1)) * (
                1.0 + v3 * (2 * psi_view[2] - 1)))
    neg = (1.0 - b) * ((1.0 + v1 * (1 - 2 * eta_view[0])) * (1.0 + v2 * (1 - 2 * eta_view[1])) * (
                1.0 + v3 * (1 - 2 * eta_view[2])))
    return pos / (pos + neg + 1e-15)


def fit_logistic_ensemble(V, p_hat, active_indices):
    """【保留原始】逻辑回归集成"""
    V_active = V[:, active_indices]
    y_pseudo = (p_hat >= 0.5).astype(int)
    if len(np.unique(y_pseudo)) == 1: return np.zeros(V.shape[1]), 0.0, False
    sample_weight = np.abs(2.0 * p_hat - 1.0) + 1e-4
    clf = LogisticRegression(fit_intercept=True, max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(V_active, y_pseudo, sample_weight=sample_weight)
    w_full = np.zeros(V.shape[1])
    w_full[active_indices] = clf.coef_[0]
    return w_full, clf.intercept_[0], True


def tie_breaking_accuracy(scores, labels):
    """【保留原始】平局处理"""
    best = np.max(scores)
    tied = np.where(np.abs(scores - best) < 1e-6)[0]
    return np.mean(labels[tied])


# ============================================================================
# 3. DCI-FUSE 评估流程（诊断版本 - 对比原始 vs 修复）
# ============================================================================
def run_dci_hle_diagnostic(data_path, K_list=[1, 2, 3, 4, 5]):
    """
    DCI-FUSE 诊断版本 - 对比两个后验概率版本的性能
    
    【诊断目的】：确定后验概率公式的修改是否导致性能下降
    """
    print("\n" + "=" * 80)
    print("DCI-FUSE [DIAGNOSTIC VERSION - Original vs Fixed Posterior]")
    print("=" * 80)

    # 加载数据
    raw_matrices, all_labels, m = load_hle_jsonl(data_path)
    norm_matrices = global_minmax_normalize(raw_matrices)
    total_qs = len(norm_matrices)

    # 【DCI创新】单题独立计算阈值
    X_local_blocks = []
    for V in norm_matrices:
        tau_q, _ = coordinate_descent_thresholds(V, max_iters=30)
        X_local_blocks.append(np.where(V >= tau_q, 1.0, -1.0))
    X_local_stacked = np.vstack(X_local_blocks)

    # 针对每个K值进行诊断
    for n_regimes in K_list:
        print(f"\n{'=' * 80}")
        print(f"K={n_regimes} Regimes")
        print('=' * 80)
        
        if n_regimes == 1:
            fuse_correct_original = 0.0
            fuse_correct_fixed = 0.0
            
            for qid, (V, labels) in enumerate(zip(norm_matrices, all_labels)):
                X_bin_q = X_local_blocks[qid]
                psi, eta, b = mom_estimates_binary(X_bin_q)
                bal_acc = (psi + eta) / 2.0
                active = [j for j in range(m) if bal_acc[j] > 0.5]

                if len(active) >= 3:
                    # 【对比】原始版本
                    p_hat_original = triplet_posterior_probabilities_original(X_bin_q, psi, eta, b, active)
                    # 【对比】修复版本
                    p_hat_fixed = triplet_posterior_probabilities_fixed(X_bin_q, psi, eta, b, active)
                else:
                    p_hat_original = p_hat_fixed = view_merging_posterior(X_bin_q, psi, eta, b, m)

                # 使用原始伪标签训练
                if len(active) >= 2:
                    w_orig, intercept_orig, ok_orig = fit_logistic_ensemble(V, p_hat_original, active)
                    fuse_scores_original = V @ w_orig + intercept_orig if ok_orig else p_hat_original
                    
                    w_fix, intercept_fix, ok_fix = fit_logistic_ensemble(V, p_hat_fixed, active)
                    fuse_scores_fixed = V @ w_fix + intercept_fix if ok_fix else p_hat_fixed
                else:
                    fuse_scores_original = p_hat_original
                    fuse_scores_fixed = p_hat_fixed

                fuse_correct_original += tie_breaking_accuracy(fuse_scores_original, labels)
                fuse_correct_fixed += tie_breaking_accuracy(fuse_scores_fixed, labels)
            
            acc_orig = fuse_correct_original / total_qs * 100
            acc_fix = fuse_correct_fixed / total_qs * 100
            delta = acc_fix - acc_orig
            
            print(f"Original Posterior:  {acc_orig:.2f}%")
            print(f"Fixed Posterior:     {acc_fix:.2f}%")
            print(f"Delta (Fixed-Original): {delta:+.2f}%")
            if delta < -0.5:
                print("⚠️  WARNING: 修复后性能下降了，原始实现可能更好")
            elif delta > 0.5:
                print("✅ 修复后性能提升了")
            else:
                print("≈  性能差异不显著")
            continue

        # K > 1：【DCI创新】多个验证体制
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

        fuse_correct_original = 0.0
        fuse_correct_fixed = 0.0
        
        for qid, (V, labels) in enumerate(zip(norm_matrices, all_labels)):
            X_bin_q = X_local_blocks[qid]
            q_regime = question_cluster_labels[qid]
            p_params = regime_parameters[q_regime]

            psi, eta, b = p_params['psi'], p_params['eta'], p_params['b']
            bal_acc = (psi + eta) / 2.0
            active = [j for j in range(m) if bal_acc[j] > 0.5]

            if len(active) >= 3:
                p_hat_original = triplet_posterior_probabilities_original(X_bin_q, psi, eta, b, active)
                p_hat_fixed = triplet_posterior_probabilities_fixed(X_bin_q, psi, eta, b, active)
            else:
                p_hat_original = p_hat_fixed = view_merging_posterior(X_bin_q, psi, eta, b, m)

            if len(active) >= 2:
                w_orig, intercept_orig, ok_orig = fit_logistic_ensemble(V, p_hat_original, active)
                fuse_scores_original = V @ w_orig + intercept_orig if ok_orig else p_hat_original
                
                w_fix, intercept_fix, ok_fix = fit_logistic_ensemble(V, p_hat_fixed, active)
                fuse_scores_fixed = V @ w_fix + intercept_fix if ok_fix else p_hat_fixed
            else:
                fuse_scores_original = p_hat_original
                fuse_scores_fixed = p_hat_fixed

            fuse_correct_original += tie_breaking_accuracy(fuse_scores_original, labels)
            fuse_correct_fixed += tie_breaking_accuracy(fuse_scores_fixed, labels)

        acc_orig = fuse_correct_original / total_qs * 100
        acc_fix = fuse_correct_fixed / total_qs * 100
        delta = acc_fix - acc_orig
        
        print(f"Original Posterior:  {acc_orig:.2f}%")
        print(f"Fixed Posterior:     {acc_fix:.2f}%")
        print(f"Delta (Fixed-Original): {delta:+.2f}%")
        if delta < -0.5:
            print("⚠️  WARNING: 修复后性能下降了，原始实现可能更好")
        elif delta > 0.5:
            print("✅ 修复后性能提升了")
        else:
            print("≈  性能差异不显著")


if __name__ == "__main__":
    run_dci_hle_diagnostic('FUSE-hle-data.jsonl')
