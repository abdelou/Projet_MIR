import argparse
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.datasets import load_cars_dataset
from src.descriptors.classical import extract_classical_feature
from src.descriptors.cnn import ResNet50Descriptor
from src.descriptors.vit import ViTB16Descriptor

CLASSICAL_DESCRIPTORS = {
    "bgr",
    "hsv",
    "hog",
    "sift",
    "orb",
    "moments",
    "lbp",
    "glcm",
}


ALL_DESCRIPTORS = [
    "bgr",
    "hsv",
    "hog",
    "sift",
    "orb",
    "moments",
    "lbp",
    "glcm",
    "resnet50",
    "vit_b_16",
]


def extract_feature(path, descriptor_name, cnn_extractor=None, vit_extractor=None):
    if descriptor_name in CLASSICAL_DESCRIPTORS:
        return extract_classical_feature(path, descriptor_name)

    if descriptor_name == "resnet50":
        return cnn_extractor.extract_one(path)

    if descriptor_name == "vit_b_16":
        return vit_extractor.extract_one(path)

    raise ValueError(f"Descripteur inconnu : {descriptor_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--descriptor",
        default="hsv",
        choices=ALL_DESCRIPTORS
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    image_paths, labels = load_cars_dataset(args.data_dir)

    features = []

    cnn_extractor = None
    vit_extractor = None

    if args.descriptor == "resnet50":
        cnn_extractor = ResNet50Descriptor()

    if args.descriptor == "vit_b_16":
        vit_extractor = ViTB16Descriptor()

    start = time.time()

    for path in tqdm(image_paths, desc=f"Extraction {args.descriptor}"):
        feat = extract_feature(
            path,
            descriptor_name=args.descriptor,
            cnn_extractor=cnn_extractor,
            vit_extractor=vit_extractor
        )
        features.append(feat)

    indexing_time = time.time() - start

    features = np.vstack(features).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    image_paths = np.asarray([str(p) for p in image_paths])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.out,
        features=features,
        labels=labels,
        image_paths=image_paths,
        descriptor=args.descriptor,
        indexing_time=indexing_time
    )

    size_mb = features.nbytes / (1024 ** 2)

    print(f"Descripteur : {args.descriptor}")
    print(f"Images indexées : {len(image_paths)}")
    print(f"Dimension descripteur : {features.shape[1]}")
    print(f"Taille features : {size_mb:.2f} MB")
    print(f"Temps indexation : {indexing_time:.2f} s")


if __name__ == "__main__":
    main()