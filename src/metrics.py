import numpy as np


def l2_normalize(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def euclidean_distance(query, database):
    query = np.asarray(query, dtype=np.float32)
    database = np.asarray(database, dtype=np.float32)
    return np.linalg.norm(database - query[None, :], axis=1)


def cosine_distance(query, database):
    query = np.asarray(query, dtype=np.float32)
    database = np.asarray(database, dtype=np.float32)

    query = query / max(np.linalg.norm(query), 1e-12)
    database = l2_normalize(database)

    similarities = database @ query
    return 1.0 - similarities


def chi_square_distance(query, database, eps=1e-10):
    query = np.maximum(np.asarray(query, dtype=np.float32), 0)
    database = np.maximum(np.asarray(database, dtype=np.float32), 0)

    numerator = (database - query[None, :]) ** 2
    denominator = database + query[None, :] + eps

    return 0.5 * np.sum(numerator / denominator, axis=1)


def bhattacharyya_distance(query, database, eps=1e-12):
    query = np.maximum(np.asarray(query, dtype=np.float32), 0)
    database = np.maximum(np.asarray(database, dtype=np.float32), 0)

    coeff = np.sum(np.sqrt(database * query[None, :]), axis=1)
    coeff = np.clip(coeff, eps, 1.0)

    return -np.log(coeff)


def correlation_distance(query, database, eps=1e-12):
    query = np.asarray(query, dtype=np.float32)
    database = np.asarray(database, dtype=np.float32)

    q = query - np.mean(query)
    X = database - np.mean(database, axis=1, keepdims=True)

    numerator = X @ q
    denominator = np.linalg.norm(X, axis=1) * np.linalg.norm(q) + eps

    corr = numerator / denominator
    return 1.0 - corr


def intersection_distance(query, database):
    query = np.maximum(np.asarray(query, dtype=np.float32), 0)
    database = np.maximum(np.asarray(database, dtype=np.float32), 0)

    intersection = np.sum(np.minimum(database, query[None, :]), axis=1)
    return 1.0 - intersection


def brute_force_distance(query, database):
    """
    Version globale vectorielle.
    Dans les TP, Brute Force était surtout utilisé pour ORB/SIFT locaux.
    Ici, avec des vecteurs fixes, cela revient à une comparaison exhaustive L2.
    """
    return euclidean_distance(query, database)


def flann_distance(query, database):
    """
    Version compatible avec le moteur vectoriel global.
    Pour une vraie version FLANN locale, il faudrait stocker les keypoints SIFT/ORB.
    """
    return euclidean_distance(query, database)


def compute_distances(query_vector, database_vectors, metric):
    metric = metric.lower()

    if metric in {"euclidean", "euclidienne", "l2"}:
        return euclidean_distance(query_vector, database_vectors)

    if metric in {"cosine", "cosinus"}:
        return cosine_distance(query_vector, database_vectors)

    if metric in {"chi_square", "chicarre", "chi-square"}:
        return chi_square_distance(query_vector, database_vectors)

    if metric == "bhattacharyya":
        return bhattacharyya_distance(query_vector, database_vectors)

    if metric == "correlation":
        return correlation_distance(query_vector, database_vectors)

    if metric == "intersection":
        return intersection_distance(query_vector, database_vectors)

    if metric in {"brute_force", "brute force", "bf"}:
        return brute_force_distance(query_vector, database_vectors)

    if metric == "flann":
        return flann_distance(query_vector, database_vectors)

    raise ValueError(f"Unknown metric: {metric}")


def rank_results(query_vector, database_vectors, metric="cosine"):
    distances = compute_distances(query_vector, database_vectors, metric)
    ranking = np.argsort(distances)
    return ranking, distances[ranking]

def compute_query_metrics(ranked_labels, query_label, total_relevant, top_values=(20, 50, 100)):
    ranked_labels = np.asarray(ranked_labels)
    relevant = (ranked_labels == query_label).astype(np.int32)

    total_relevant = int(total_relevant)
    if total_relevant <= 0:
        raise ValueError("total_relevant must be > 0")

    out = {}

    for k in top_values:
        k = int(k)
        rel_k = relevant[:k]
        tp_k = int(rel_k.sum())

        out[f"P@{k}"] = tp_k / k
        out[f"R@{k}"] = min(tp_k, total_relevant) / total_relevant

        cum_tp = np.cumsum(rel_k)
        ranks = np.arange(1, len(rel_k) + 1)
        precisions_at_hits = (cum_tp / ranks) * rel_k

        denom = min(total_relevant, k)
        out[f"AP@{k}"] = float(precisions_at_hits.sum() / denom) if denom > 0 else 0.0

    r = min(total_relevant, len(relevant))
    out["R-Precision"] = float(relevant[:r].sum() / r) if r > 0 else 0.0

    cum_tp_full = np.cumsum(relevant)
    ranks_full = np.arange(1, len(relevant) + 1)
    precisions_full = (cum_tp_full / ranks_full) * relevant
    out["AP_full"] = float(precisions_full.sum() / total_relevant)

    return out