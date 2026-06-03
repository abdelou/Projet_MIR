import argparse
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import GROUP_08_QUERIES
from src.datasets import find_query_image
from src.search import rank_results
from src.metrics import compute_query_metrics
from src.visualization import save_topk_grid


HISTOGRAM_DESCRIPTORS = {"bgr", "hsv", "lbp"}
DEEP_DESCRIPTORS = {"resnet50", "vit_b_16"}
L2_DESCRIPTORS = {"hog", "sift", "orb", "moments", "glcm"}


def choose_default_metric(descriptor_name):
    descriptor_name = descriptor_name.lower()

    if descriptor_name in HISTOGRAM_DESCRIPTORS:
        return "chi_square"

    if descriptor_name in DEEP_DESCRIPTORS:
        return "cosine"

    if descriptor_name in L2_DESCRIPTORS:
        return "euclidean"

    return "cosine"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--metric", default="auto")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--visual-dir", default=None)

    args = parser.parse_args()

    data = np.load(args.index, allow_pickle=True)

    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    image_paths = data["image_paths"]

    descriptor_name = str(data["descriptor"]) if "descriptor" in data else Path(args.index).stem
    metric = choose_default_metric(descriptor_name) if args.metric == "auto" else args.metric

    rows = []
    search_times = []

    for query_id, query_class, query_stem in GROUP_08_QUERIES:
        query_path = find_query_image(image_paths, query_stem)
        query_index = list(image_paths).index(str(query_path))
        query_vector = features[query_index]

        start = time.perf_counter()
        ranking, _ = rank_results(query_vector, features, metric=metric)
        search_time = time.perf_counter() - start
        search_times.append(search_time)

        ranking = np.asarray([idx for idx in ranking if idx != query_index], dtype=np.int64)

        ranked_labels = labels[ranking]
        total_relevant = int(np.sum(labels == query_class)) - 1

        metrics = compute_query_metrics(
            ranked_labels=ranked_labels,
            query_label=query_class,
            total_relevant=total_relevant,
            top_values=(20, 50, 100),
        )

        if args.visual_dir is not None:
            retrieved_paths = image_paths[ranking]
            retrieved_labels = labels[ranking]
            out_dir = Path(args.visual_dir) / f"{descriptor_name}_{metric}"

            save_topk_grid(
                query_path=image_paths[query_index],
                retrieved_paths=retrieved_paths,
                retrieved_labels=retrieved_labels,
                query_label=query_class,
                out_path=out_dir / f"{query_id}_top20.png",
                k=20,
            )

            save_topk_grid(
                query_path=image_paths[query_index],
                retrieved_paths=retrieved_paths,
                retrieved_labels=retrieved_labels,
                query_label=query_class,
                out_path=out_dir / f"{query_id}_top50.png",
                k=50,
            )

        row = {
            "query": query_id,
            "query_class": query_class,
            "query_image": query_stem,
            "descriptor": descriptor_name,
            "metric": metric,
            "total_relevant": total_relevant,
            "search_time_s": search_time,
        }

        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    for col in ["AP@20", "AP@50", "AP@100", "AP_full"]:
        df["m" + col] = df[col].mean()

    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.results_csv, index=False)

    summary = {
        "descriptor": descriptor_name,
        "metric": metric,
        "n_images": int(len(image_paths)),
        "descriptor_dim": int(features.shape[1]),
        "features_size_mb": float(features.nbytes / (1024 ** 2)),
        "indexing_time_s": float(data["indexing_time"]) if "indexing_time" in data else None,
        "avg_search_time_per_query_s": float(np.mean(search_times)),
        "avg_search_time_per_db_image_s": float(np.mean(search_times) / len(image_paths)),
        "mean_P@20": float(df["P@20"].mean()),
        "mean_P@50": float(df["P@50"].mean()),
        "mean_P@100": float(df["P@100"].mean()),
        "mAP@20": float(df["AP@20"].mean()),
        "mAP@50": float(df["AP@50"].mean()),
        "mAP@100": float(df["AP@100"].mean()),
        "mAP_full": float(df["AP_full"].mean()),
        "mean_R_Precision": float(df["R-Precision"].mean()),
    }

    if args.summary_json is not None:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(df)
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()