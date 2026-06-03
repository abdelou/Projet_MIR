from pathlib import Path
import time
import numpy as np

from src.search import compute_distances
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

HISTOGRAM_DESCRIPTORS = {
    "bgr",
    "hsv",
    "lbp",
}

L2_DESCRIPTORS = {
    "hog",
    "sift",
    "orb",
    "moments",
    "glcm",
}

DEEP_DESCRIPTORS = {
    "resnet50",
    "vit_b_16",
}


class MultiDescriptorRetrievalEngine:
    def __init__(self, descriptor_index_paths):
        self.descriptor_index_paths = {
            name: Path(path)
            for name, path in descriptor_index_paths.items()
        }

        self.indexes = {}
        self.extractors = {}

        for descriptor_name, index_path in self.descriptor_index_paths.items():
            if not index_path.exists():
                raise FileNotFoundError(
                    f"Index file not found for {descriptor_name}: {index_path}"
                )

            data = np.load(index_path, allow_pickle=True)

            self.indexes[descriptor_name] = {
                "features": data["features"].astype(np.float32),
                "labels": data["labels"].astype(np.int64),
                "image_paths": data["image_paths"],
            }

        first_descriptor = next(iter(self.indexes))
        self.image_paths = self.indexes[first_descriptor]["image_paths"]
        self.labels = self.indexes[first_descriptor]["labels"]

        # Vérification importante : tous les index doivent être alignés.
        for descriptor_name, index_data in self.indexes.items():
            if not np.array_equal(index_data["image_paths"], self.image_paths):
                raise ValueError(
                    f"Index alignment error for {descriptor_name}: image_paths differ. "
                    "Regenerate all descriptor indexes from the same data/cars directory."
                )

            if not np.array_equal(index_data["labels"], self.labels):
                raise ValueError(
                    f"Index alignment error for {descriptor_name}: labels differ. "
                    "Regenerate all descriptor indexes from the same data/cars directory."
                )

        if "resnet50" in self.indexes:
            self.extractors["resnet50"] = ResNet50Descriptor()

        if "vit_b_16" in self.indexes:
            self.extractors["vit_b_16"] = ViTB16Descriptor()

    def extract_query_feature(self, descriptor_name, query_image_path):
        descriptor_name = descriptor_name.lower()

        if descriptor_name in CLASSICAL_DESCRIPTORS:
            return extract_classical_feature(
                query_image_path,
                descriptor_name
            ).astype(np.float32)

        if descriptor_name == "resnet50":
            return self.extractors["resnet50"].extract_one(query_image_path).astype(np.float32)

        if descriptor_name == "vit_b_16":
            return self.extractors["vit_b_16"].extract_one(query_image_path).astype(np.float32)

        raise ValueError(f"Unsupported descriptor: {descriptor_name}")

    def choose_metric(self, descriptor_name, requested_metric):
        descriptor_name = descriptor_name.lower()

        if requested_metric != "auto":
            return requested_metric

        if descriptor_name in HISTOGRAM_DESCRIPTORS:
            return "chi_square"

        if descriptor_name in L2_DESCRIPTORS:
            return "euclidean"

        if descriptor_name in DEEP_DESCRIPTORS:
            return "cosine"

        return "cosine"

    @staticmethod
    def normalize_distances(distances, eps=1e-12):
        distances = distances.astype(np.float32)

        d_min = float(np.min(distances))
        d_max = float(np.max(distances))

        if abs(d_max - d_min) < eps:
            return np.zeros_like(distances, dtype=np.float32)

        return (distances - d_min) / (d_max - d_min)

    def search(self, query_image_path, selected_descriptors, metric="auto", top_k=50):
        query_image_path = Path(query_image_path)
        query_stem = query_image_path.stem

        if not selected_descriptors:
            raise ValueError("At least one descriptor must be selected.")

        start = time.time()

        normalized_distance_vectors = []
        used_metrics = {}

        for descriptor_name in selected_descriptors:
            descriptor_name = descriptor_name.lower()

            if descriptor_name not in self.indexes:
                raise ValueError(f"Descriptor index not loaded: {descriptor_name}")

            query_vector = self.extract_query_feature(
                descriptor_name=descriptor_name,
                query_image_path=query_image_path,
            )

            database_vectors = self.indexes[descriptor_name]["features"]

            if query_vector.shape[0] != database_vectors.shape[1]:
                raise ValueError(
                    f"Dimension mismatch for descriptor '{descriptor_name}': "
                    f"query has dimension {query_vector.shape[0]}, "
                    f"database has dimension {database_vectors.shape[1]}. "
                    f"Regenerate the .npz index for this descriptor."
                )

            effective_metric = self.choose_metric(descriptor_name, metric)
            used_metrics[descriptor_name] = effective_metric

            distances = compute_distances(
                query_vector=query_vector,
                database_vectors=database_vectors,
                metric=effective_metric,
            )

            normalized_distances = self.normalize_distances(distances)
            normalized_distance_vectors.append(normalized_distances)

        fused_distances = np.mean(np.vstack(normalized_distance_vectors), axis=0)
        ranking = np.argsort(fused_distances)

        search_time = time.time() - start

        results = []

        for db_index in ranking:
            db_path = Path(str(self.image_paths[db_index]))

            # Si la requête vient de la base, on ne retourne pas l'image elle-même.
            if db_path.stem == query_stem:
                continue

            results.append({
                "rank": len(results) + 1,
                "image_path": str(db_path),
                "label": int(self.labels[db_index]),
                "distance": float(fused_distances[db_index]),
            })

            if len(results) >= top_k:
                break

        return results, search_time, used_metrics