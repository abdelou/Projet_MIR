from pathlib import Path
import re

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root_dir):
    root_dir = Path(root_dir)

    image_paths = [
        p for p in root_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_paths = sorted(image_paths)
    return image_paths


def extract_class_from_filename(path):
    """
    Exemple :
    1_4_Kia_stinger_1990.jpg -> classe 1
    """
    name = Path(path).name
    match = re.match(r"^(\d+)_", name)

    if match is None:
        raise ValueError(f"Impossible d'extraire la classe depuis : {name}")

    return int(match.group(1))


def load_cars_dataset(cars_dir):
    image_paths = list_images(cars_dir)
    labels = [extract_class_from_filename(p) for p in image_paths]

    return image_paths, labels


def find_query_image(image_paths, query_stem):
    """
    Cherche une image dont le nom commence par query_stem.
    Exemple :
    query_stem = '1_4_Kia_stinger_1990'
    match = '1_4_Kia_stinger_1990.jpg'
    """
    matches = [
        p for p in image_paths
        if Path(p).stem == query_stem or Path(p).stem.startswith(query_stem)
    ]

    if len(matches) == 0:
        raise FileNotFoundError(f"Image requête introuvable : {query_stem}")

    if len(matches) > 1:
        print(f"Attention : plusieurs matches pour {query_stem}, on prend le premier.")

    return matches[0]