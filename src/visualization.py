import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def save_topk_grid(query_path, retrieved_paths, retrieved_labels, query_label, out_path, k=20):
    retrieved_paths = retrieved_paths[:k]
    retrieved_labels = retrieved_labels[:k]

    cols = 5
    rows = int((k + cols - 1) / cols)

    plt.figure(figsize=(15, 3 * rows))

    for i, path in enumerate(retrieved_paths):
        img = Image.open(path).convert("RGB")

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")

        label = retrieved_labels[i]
        ok = "OK" if int(label) == int(query_label) else "NO"
        plt.title(f"{i+1} | c={label} | {ok}", fontsize=8)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_rp_curve(recalls, precisions, out_path, title="Recall/Precision curve"):
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()