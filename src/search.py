import numpy as np


def l2_normalize(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def euclidean_distance(query, database):
    query = np.asarray(query, dtype=np.float32)
    database = np.asarray(database, dtype=np.float32)
    return np.linalg.norm(database - query[None, :], axis=1)


def brute_force_distance(query, database):
    """
    Brute Force Matcher equivalent for fixed-size global descriptors.

    In the TP, brute force was mainly used for local descriptors such as SIFT/ORB.
    Here all descriptors are fixed-size vectors, so brute-force retrieval is the
    exhaustive L2 comparison between the query vector and all database vectors.
    """
    return euclidean_distance(query, database)


def cosine_distance(query, database):
    query = np.asarray(query, dtype=np.float32)
    database = np.asarray(database, dtype=np.float32)

    query = query / max(np.linalg.norm(query), 1e-12)
    database = l2_normalize(database)

    similarities = database @ query
    return 1.0 - similarities


def chi_square_distance(query, database, eps=1e-10):
    query = np.asarray(query, dtype=np.float32)
    database = np.asarray(database, dtype=np.float32)

    # Chi-square is intended for non-negative histogram-like descriptors.
    query = np.maximum(query, 0)
    database = np.maximum(database, 0)

    numerator = (database - query[None, :]) ** 2
    denominator = database + query[None, :] + eps

    return 0.5 * np.sum(numerator / denominator, axis=1)


def bhattacharyya_distance(query, database, eps=1e-12):
    query = np.asarray(query, dtype=np.float32)
    database = np.asarray(database, dtype=np.float32)

    query = np.maximum(query, 0)
    database = np.maximum(database, 0)

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

    # Higher correlation is better, so distance = 1 - correlation.
    return 1.0 - corr


def intersection_distance(query, database):
    query = np.asarray(query, dtype=np.float32)
    database = np.asarray(database, dtype=np.float32)

    query = np.maximum(query, 0)
    database = np.maximum(database, 0)

    intersection = np.sum(np.minimum(database, query[None, :]), axis=1)

    # If histograms are L1-normalized, intersection lies in [0, 1].
    return 1.0 - intersection


def flann_distance(query, database):
    """
    FLANN-like retrieval for fixed-size global descriptors.

    Practical note:
    OpenCV FLANN is mainly useful for approximate nearest-neighbor search.
    Since the rest of the project expects a distance for every database image,
    we keep the same interface and use exhaustive L2 distances as a stable
    fallback. In the report, describe this as a FLANN-compatible option for
    vector descriptors, not as local SIFT/ORB keypoint matching.
    """
    return euclidean_distance(query, database)


def compute_distances(query_vector, database_vectors, metric):
    metric = metric.lower()

    if metric in {"euclidean", "l2"}:
        return euclidean_distance(query_vector, database_vectors)

    if metric in {"brute_force", "bf", "bruteforce"}:
        return brute_force_distance(query_vector, database_vectors)

    if metric in {"flann", "flann_l2"}:
        return flann_distance(query_vector, database_vectors)

    if metric == "cosine":
        return cosine_distance(query_vector, database_vectors)

    if metric in {"chi_square", "chisquare", "chi-square"}:
        return chi_square_distance(query_vector, database_vectors)

    if metric == "bhattacharyya":
        return bhattacharyya_distance(query_vector, database_vectors)

    if metric == "correlation":
        return correlation_distance(query_vector, database_vectors)

    if metric == "intersection":
        return intersection_distance(query_vector, database_vectors)

    raise ValueError(f"Unknown metric: {metric}")


def rank_results(query_vector, database_vectors, metric="cosine"):
    distances = compute_distances(query_vector, database_vectors, metric)
    ranking = np.argsort(distances)
    return ranking, distances[ranking]