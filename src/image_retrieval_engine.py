from pathlib import Path
import time
import numpy as np

from src.search import rank_results
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

class ImageRetrievalEngine:
    """
    Content-Based Image Retrieval engine.

    Offline phase:
        descriptors are extracted and stored by part1_index_cars.py.

    Online phase:
        a query image is represented with the same descriptor, compared
        with the database descriptors, and the Top-K most similar images
        are returned.
    """

    def __init__(self, index_path):
        self.index_path = Path(index_path)
        self.data = np.load(self.index_path, allow_pickle=True)

        self.features = self.data["features"].astype(np.float32)
        self.labels = self.data["labels"].astype(np.int64)
        self.image_paths = self.data["image_paths"]

        if "descriptor" in self.data:
            self.descriptor = str(self.data["descriptor"])
        else:
            self.descriptor = "unknown"

        self.extractor = None

        if self.descriptor == "resnet50":
            self.extractor = ResNet50Descriptor()

        elif self.descriptor == "vit_b_16":
            self.extractor = ViTB16Descriptor()

    def extract_query_feature(self, query_image_path):
        if self.descriptor in CLASSICAL_DESCRIPTORS:
            return extract_classical_feature(query_image_path, self.descriptor).astype(np.float32)

        if self.descriptor == "resnet50":
            return self.extractor.extract_one(query_image_path).astype(np.float32)

        if self.descriptor == "vit_b_16":
            return self.extractor.extract_one(query_image_path).astype(np.float32)

        raise ValueError(f"Unsupported descriptor: {self.descriptor}")

    def search(self, query_image_path, metric="cosine", top_k=50, exclude_query_if_present=True):
        query_image_path = Path(query_image_path)
        query_vector = self.extract_query_feature(query_image_path)

        start = time.time()
        ranking, sorted_distances = rank_results(
            query_vector=query_vector,
            database_vectors=self.features,
            metric=metric
        )
        search_time = time.time() - start

        results = []

        for rank_idx, db_index in enumerate(ranking):
            db_path = Path(str(self.image_paths[db_index]))

            if exclude_query_if_present:
                try:
                    if db_path.resolve() == query_image_path.resolve():
                        continue
                except FileNotFoundError:
                    pass

            results.append({
                "rank": len(results) + 1,
                "image_path": str(db_path),
                "label": int(self.labels[db_index]),
                "distance": float(sorted_distances[rank_idx])
            })

            if len(results) >= top_k:
                break

        return results, search_time