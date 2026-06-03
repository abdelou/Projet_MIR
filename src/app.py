from pathlib import Path
import shutil
import base64
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from src.multi_retrieval_engine import MultiDescriptorRetrievalEngine


BASE_DIR = Path(__file__).resolve().parents[1]

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "src" / "templates"),
    static_folder=str(BASE_DIR / "static")
)


DESCRIPTOR_INDEXES = {
    "bgr": BASE_DIR / "artifacts" / "descriptors" / "cars_bgr.npz",
    "hsv": BASE_DIR / "artifacts" / "descriptors" / "cars_hsv.npz",
    "hog": BASE_DIR / "artifacts" / "descriptors" / "cars_hog.npz",
    "sift": BASE_DIR / "artifacts" / "descriptors" / "cars_sift.npz",
    "orb": BASE_DIR / "artifacts" / "descriptors" / "cars_orb.npz",
    "moments": BASE_DIR / "artifacts" / "descriptors" / "cars_moments.npz",
    "lbp": BASE_DIR / "artifacts" / "descriptors" / "cars_lbp.npz",
    "glcm": BASE_DIR / "artifacts" / "descriptors" / "cars_glcm.npz",
    "resnet50": BASE_DIR / "artifacts" / "descriptors" / "cars_resnet50.npz",
    "vit_b_16": BASE_DIR / "artifacts" / "descriptors" / "cars_vit_b_16.npz",
}

LOADED_ENGINE = None
LOADED_DESCRIPTORS = []


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/load_descriptors", methods=["POST"])
def load_descriptors():
    global LOADED_ENGINE, LOADED_DESCRIPTORS

    selected_descriptors = request.form.getlist("descriptors")

    if not selected_descriptors:
        return jsonify({"error": "No descriptor selected."}), 400

    selected_index_paths = {}

    for descriptor in selected_descriptors:
        if descriptor not in DESCRIPTOR_INDEXES:
            return jsonify({"error": f"Unknown descriptor: {descriptor}"}), 400

        index_path = DESCRIPTOR_INDEXES[descriptor]

        if not index_path.exists():
            return jsonify({
                "error": f"Missing index for {descriptor}: {index_path}"
            }), 400

        selected_index_paths[descriptor] = index_path

    LOADED_ENGINE = MultiDescriptorRetrievalEngine(selected_index_paths)
    LOADED_DESCRIPTORS = selected_descriptors

    return jsonify({
        "message": "Descriptors loaded successfully.",
        "loaded_descriptors": LOADED_DESCRIPTORS
    })


@app.route("/search_unimodal", methods=["POST"])
def search_unimodal():
    global LOADED_ENGINE, LOADED_DESCRIPTORS

    uploaded_file = request.files.get("image")

    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "No query image uploaded."}), 400

    selected_descriptors = request.form.getlist("descriptors")
    distance = request.form.get("distance", "auto")
    top_k = int(request.form.get("top_k", 50))

    if not selected_descriptors:
        return jsonify({"error": "No descriptor selected."}), 400

    if LOADED_ENGINE is None:
        return jsonify({
            "error": "Descriptors are not loaded. Click 'Load descriptors' first."
        }), 400

    if set(selected_descriptors) != set(LOADED_DESCRIPTORS):
        return jsonify({
            "error": "Selected descriptors differ from loaded descriptors. Click 'Load descriptors' again."
        }), 400

    filename = secure_filename(uploaded_file.filename)
    query_path = UPLOAD_DIR / filename
    uploaded_file.save(query_path)

    results, search_time, used_metrics = LOADED_ENGINE.search(
        query_image_path=query_path,
        selected_descriptors=selected_descriptors,
        metric=distance,
        top_k=top_k
    )

    for old_file in RESULTS_DIR.glob("*"):
        if old_file.is_file():
            old_file.unlink()

    json_results = []

    for r in results:
        src_path = Path(r["image_path"])

        if not src_path.exists():
            continue

        dst_path = RESULTS_DIR / src_path.name
        shutil.copy(src_path, dst_path)

        json_results.append({
            "rank": r["rank"],
            "class": r["label"],
            "score": r["distance"],
            "path": url_for("static", filename=f"results/{dst_path.name}")
        })

    pr_curve, metrics = make_pr_curve_and_metrics(
        query_path=query_path,
        results=results,
        engine=LOADED_ENGINE,
        k=top_k
    )

    return jsonify({
        "results": json_results,
        "search_time": search_time,
        "used_metrics": used_metrics,
        "pr_curve": pr_curve,
        "metrics": metrics
    })


def extract_query_label(query_path):
    try:
        return int(Path(query_path).stem.split("_")[0])
    except Exception:
        return None


def make_pr_curve_and_metrics(query_path, results, engine, k):
    query_label = extract_query_label(query_path)

    if query_label is None:
        return None, {
            "message": "Cannot compute R/P curve: query class could not be inferred from filename."
        }

    labels = np.asarray(engine.labels)
    total_relevant = int(np.sum(labels == query_label))

    query_stem = Path(query_path).stem
    if any(Path(str(p)).stem == query_stem for p in engine.image_paths):
        total_relevant -= 1

    if total_relevant <= 0:
        return None, {
            "message": "Cannot compute R/P curve: no relevant images found."
        }

    precisions = []
    recalls = []
    tp = 0

    for i, r in enumerate(results, start=1):
        if int(r["label"]) == int(query_label):
            tp += 1

        precisions.append(tp / i)
        recalls.append(min(tp / total_relevant, 1.0))

    top_results = results[:k]
    relevant_flags = [int(r["label"]) == int(query_label) for r in top_results]

    denom_precision = max(1, len(top_results))
    p_at_k = sum(relevant_flags) / denom_precision
    r_at_k = min(sum(relevant_flags), total_relevant) / total_relevant

    ap_values = []
    tp_ap = 0

    for i, rel in enumerate(relevant_flags, start=1):
        if rel:
            tp_ap += 1
            ap_values.append(tp_ap / i)

    denom = min(total_relevant, k)
    ap_at_k = float(sum(ap_values) / denom) if denom > 0 else 0.0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        [r * 100 for r in recalls],
        [p * 100 for p in precisions],
        marker="o"
    )
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title(f"Recall/Precision curve - class {query_label}")
    ax.grid(True)

    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)

    buffer.seek(0)
    pr_curve_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    metrics = {
        "query_class": query_label,
        "top_k": k,
        "total_relevant": total_relevant,
        f"P@{k}": round(p_at_k, 4),
        f"R@{k}": round(r_at_k, 4),
        f"AP@{k}": round(ap_at_k, 4),
    }

    return pr_curve_base64, metrics


@app.route("/search_multimodal", methods=["POST"])
def search_multimodal():
    return jsonify({
        "error": "Multimodal search is not connected to the Flask interface yet."
    }), 501


if __name__ == "__main__":
    app.run(debug=True, port=5000)