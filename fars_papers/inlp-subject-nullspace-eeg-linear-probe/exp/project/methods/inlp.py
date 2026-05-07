# Iterative Nullspace Projection (INLP) for multinomial subject-ID removal.
# Adapted from https://github.com/shauli-ravfogel/nullspace_projection
# Memory-efficient implementation: stores basis vectors (not d x d matrices)
# to handle 17600-dim EEG embeddings. Uses Ben-Israel formula for final
# projection via re-orthonormalization of accumulated rowspace bases.
# Uses SGDClassifier(loss='log_loss') instead of LogisticRegression(solver='lbfgs')
# for computational feasibility on 17600-dim data (~30s vs 20min+ per fit).

import warnings
import numpy as np
import scipy.linalg
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def _get_rowspace_basis(W):
    if np.allclose(W, 0):
        return np.zeros((W.shape[1], 0))
    basis = scipy.linalg.orth(W.T)
    return basis


def apply_projection(Z, V):
    if V.shape[1] == 0:
        return Z.copy()
    return Z - (Z @ V) @ V.T


def run_inlp(Z_train, subject_ids_train, max_iter=10, early_stop_threshold=1.25, seed=42):
    n_samples, d = Z_train.shape
    subjects_unique = np.unique(subject_ids_train)
    S = len(subjects_unique)
    chance = 1.0 / S
    threshold = early_stop_threshold * chance

    Z_inlp_train, Z_inlp_val, y_inlp_train, y_inlp_val = train_test_split(
        Z_train, subject_ids_train,
        test_size=0.2, stratify=subject_ids_train, random_state=seed
    )

    clf_pre = SGDClassifier(
        loss="log_loss", max_iter=1000, random_state=seed, tol=1e-3
    )
    clf_pre.fit(Z_inlp_train, y_inlp_train)
    pre_inlp_acc = accuracy_score(y_inlp_val, clf_pre.predict(Z_inlp_val))

    all_basis_vectors = []
    iter_log = []
    num_iterations = 0

    Z_train_orig = Z_inlp_train.copy()
    Z_val_orig = Z_inlp_val.copy()

    Z_t_train = Z_inlp_train.copy()
    Z_t_val = Z_inlp_val.copy()

    for t in range(max_iter):
        clf = SGDClassifier(
            loss="log_loss", max_iter=1000, random_state=seed, tol=1e-3
        )
        clf.fit(Z_t_train, y_inlp_train)

        train_acc = accuracy_score(y_inlp_train, clf.predict(Z_t_train))
        val_acc = accuracy_score(y_inlp_val, clf.predict(Z_t_val))

        iter_log.append({
            "iteration": t,
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
        })

        if val_acc <= threshold:
            break

        W_t = clf.coef_
        basis = _get_rowspace_basis(W_t)

        if basis.shape[1] == 0:
            break

        all_basis_vectors.append(basis)
        num_iterations = t + 1

        B_so_far = np.hstack(all_basis_vectors)
        V_so_far = scipy.linalg.orth(B_so_far)
        Z_t_train = apply_projection(Z_train_orig, V_so_far)
        Z_t_val = apply_projection(Z_val_orig, V_so_far)

    if len(all_basis_vectors) == 0:
        V_final = np.zeros((d, 0))
    else:
        B_all = np.hstack(all_basis_vectors)
        V_final = scipy.linalg.orth(B_all)

    total_basis_count = sum(b.shape[1] for b in all_basis_vectors)
    assert total_basis_count <= S * max_iter, (
        f"Basis count {total_basis_count} exceeds S*max_iter={S * max_iter}"
    )

    Z_proj_check = apply_projection(Z_inlp_train[:1], V_final)
    assert Z_proj_check.shape == Z_inlp_train[:1].shape, (
        f"Shape mismatch: {Z_proj_check.shape} vs {Z_inlp_train[:1].shape}"
    )

    if len(iter_log) > 1:
        val_accs = [e["val_acc"] for e in iter_log]
        for i in range(1, len(val_accs)):
            if val_accs[i] > val_accs[i - 1] + 0.02:
                warnings.warn(
                    f"INLP: val accuracy increased at iter {i}: "
                    f"{val_accs[i-1]:.4f} -> {val_accs[i]:.4f}"
                )

    Z_post_train = apply_projection(Z_inlp_train, V_final)
    Z_post_val = apply_projection(Z_inlp_val, V_final)
    clf_post = SGDClassifier(
        loss="log_loss", max_iter=1000, random_state=seed, tol=1e-3
    )
    clf_post.fit(Z_post_train, y_inlp_train)
    post_inlp_acc = accuracy_score(y_inlp_val, clf_post.predict(Z_post_val))

    rank_removed = V_final.shape[1]

    return {
        "V": V_final,
        "num_iterations": num_iterations,
        "rank_removed": rank_removed,
        "pre_inlp_subject_acc": float(pre_inlp_acc),
        "post_inlp_subject_acc": float(post_inlp_acc),
        "iter_log": iter_log,
        "chance_level": float(chance),
        "threshold": float(threshold),
    }


def run_inlp_progressive(Z_train, subject_ids_train, max_iter=10, seed=42):
    n_samples, d = Z_train.shape
    subjects_unique = np.unique(subject_ids_train)
    S = len(subjects_unique)

    Z_inlp_train, Z_inlp_val, y_inlp_train, y_inlp_val = train_test_split(
        Z_train, subject_ids_train,
        test_size=0.2, stratify=subject_ids_train, random_state=seed
    )

    clf_pre = SGDClassifier(
        loss="log_loss", max_iter=1000, random_state=seed, tol=1e-3
    )
    clf_pre.fit(Z_inlp_train, y_inlp_train)
    pre_inlp_acc = accuracy_score(y_inlp_val, clf_pre.predict(Z_inlp_val))

    all_basis_vectors = []
    iter_log = []
    V_per_iter = {}

    Z_train_orig = Z_inlp_train.copy()
    Z_val_orig = Z_inlp_val.copy()
    Z_t_train = Z_inlp_train.copy()
    Z_t_val = Z_inlp_val.copy()

    for t in range(max_iter):
        clf = SGDClassifier(
            loss="log_loss", max_iter=1000, random_state=seed, tol=1e-3
        )
        clf.fit(Z_t_train, y_inlp_train)

        train_acc = accuracy_score(y_inlp_train, clf.predict(Z_t_train))
        val_acc = accuracy_score(y_inlp_val, clf.predict(Z_t_val))

        iter_log.append({
            "iteration": t,
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
        })

        W_t = clf.coef_
        basis = _get_rowspace_basis(W_t)
        if basis.shape[1] == 0:
            break

        all_basis_vectors.append(basis)

        B_so_far = np.hstack(all_basis_vectors)
        V_so_far = scipy.linalg.orth(B_so_far)
        V_per_iter[t + 1] = V_so_far

        Z_t_train = apply_projection(Z_train_orig, V_so_far)
        Z_t_val = apply_projection(Z_val_orig, V_so_far)

    return {
        "V_per_iter": V_per_iter,
        "pre_inlp_subject_acc": float(pre_inlp_acc),
        "iter_log": iter_log,
        "chance_level": 1.0 / S,
    }
