import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops


def _read_bgr(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    return image


def _l1_normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float32).ravel()
    s = np.sum(np.abs(v))
    return v / (s + eps)


def _l2_normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float32).ravel()
    n = np.linalg.norm(v)
    return v / max(n, eps)


def extract_bgr_histogram(image_path):
    """
    BGR histogram descriptor, as in TP2/TP3.
    Output dimension: 768 = 256 B + 256 G + 256 R.
    """
    image = _read_bgr(image_path)

    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

    feat = np.concatenate([hist_b.ravel(), hist_g.ravel(), hist_r.ravel()])
    return _l1_normalize(feat)


def extract_hsv_histogram(image_path):
    """
    HSV histogram descriptor, close to the TP version.
    Output dimension: 540 = 180 H + 180 S + 180 V.
    """
    image = _read_bgr(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [180], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [180], [0, 256])

    feat = np.concatenate([hist_h.ravel(), hist_s.ravel(), hist_v.ravel()])
    return _l1_normalize(feat)


def extract_hog_descriptor(image_path):
    """
    HOG descriptor.
    We use a fixed image size so that all images have the same descriptor size.
    """
    image = _read_bgr(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )

    feat = hog.compute(gray)
    return _l2_normalize(feat)


def extract_lbp_descriptor(image_path):
    """
    LBP texture histogram.
    Output dimension: 256.
    """
    image = _read_bgr(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    P = 8
    R = 1
    lbp = local_binary_pattern(gray, P, R, method="default")

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return _l1_normalize(hist)


def extract_glcm_descriptor(image_path):
    """
    GLCM texture descriptor.
    Output dimension: 7.
    """
    image = _read_bgr(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    levels = 16
    q = (gray.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    q[q >= levels] = levels - 1

    glcm = graycomatrix(
        q,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True
    )

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

    feat = []

    for prop in props:
        value = graycoprops(glcm, prop)
        feat.append(float(np.mean(value)))

    P = glcm.astype(np.float64)
    entropy = -np.sum(P * np.log(P + 1e-12))
    feat.append(float(entropy))

    return _l2_normalize(np.asarray(feat, dtype=np.float32))


def extract_hu_moments_descriptor(image_path):
    """
    Hu moments descriptor.
    This corresponds to the 'Mom.' checkbox from the TP-style interface.
    Output dimension: 7.
    """
    image = _read_bgr(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()

    # Log transform for numerical stability.
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return _l2_normalize(hu)


def extract_sift_pooled_descriptor(image_path, max_features=500):
    """
    SIFT local descriptors converted into one fixed-size global vector.

    TP SIFT stores a variable-size matrix of local descriptors.
    For the Flask multi-descriptor engine, we need a fixed-size vector.
    We therefore use mean + standard deviation pooling.

    Output dimension: 256 = 128 mean + 128 std.
    """
    image = _read_bgr(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) == 0:
        return np.zeros(256, dtype=np.float32)

    descriptors = descriptors.astype(np.float32)

    mean = np.mean(descriptors, axis=0)
    std = np.std(descriptors, axis=0)

    feat = np.concatenate([mean, std])
    return _l2_normalize(feat)


def extract_orb_pooled_descriptor(image_path, max_features=500):
    """
    ORB local descriptors converted into one fixed-size global vector.

    TP ORB stores a variable-size matrix of local binary descriptors.
    For the Flask multi-descriptor engine, we use mean + std pooling.

    Output dimension: 64 = 32 mean + 32 std.
    """
    image = _read_bgr(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) == 0:
        return np.zeros(64, dtype=np.float32)

    descriptors = descriptors.astype(np.float32)

    mean = np.mean(descriptors, axis=0)
    std = np.std(descriptors, axis=0)

    feat = np.concatenate([mean, std])
    return _l2_normalize(feat)


def extract_classical_feature(image_path, descriptor_name):
    """
    Unified entry point for all TP-style classical descriptors.
    """
    descriptor_name = descriptor_name.lower()

    if descriptor_name == "bgr":
        return extract_bgr_histogram(image_path)

    if descriptor_name == "hsv":
        return extract_hsv_histogram(image_path)

    if descriptor_name == "hog":
        return extract_hog_descriptor(image_path)

    if descriptor_name == "sift":
        return extract_sift_pooled_descriptor(image_path)

    if descriptor_name == "orb":
        return extract_orb_pooled_descriptor(image_path)

    if descriptor_name == "moments":
        return extract_hu_moments_descriptor(image_path)

    if descriptor_name == "lbp":
        return extract_lbp_descriptor(image_path)

    if descriptor_name == "glcm":
        return extract_glcm_descriptor(image_path)

    raise ValueError(f"Descripteur classique inconnu : {descriptor_name}")