"""
PhotoMap pipeline.

Section 3 (représentations) : implémentée (DINOv2 ou ResNet-18).
Sections 4, 5, 6 (projection, voisins, groupes) : placeholders aléatoires
conservés pour que l'application continue de fonctionner.
"""

from pathlib import Path
import random

import torch
from PIL import Image
from torch.nn.functional import normalize
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# "dinov2" (auto-supervisé, ViT-S/14, d=384) ou "resnet18" (supervisé, d=512)
MODEL_NAME = "dinov2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_ENCODERS = {}  # cache mémoire : nom -> (encodeur, preprocess, dimension)


# ---------------------------------------------------------------------------
# Collection de photos
# ---------------------------------------------------------------------------

def get_photos(photo_dir):
    """Return the images available in the real photo collection."""

    photo_dir = Path(photo_dir)
    photo_dir.mkdir(parents=True, exist_ok=True)

    return [
        path
        for path in sorted(photo_dir.iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ---------------------------------------------------------------------------
# Section 3 : représentation de la collection
# ---------------------------------------------------------------------------

def _load_encoder(name=None):
    """Charge l'encodeur une seule fois et retourne (modèle, preprocess, d)."""

    name = name or MODEL_NAME

    if name in _ENCODERS:
        return _ENCODERS[name]

    if name == "dinov2":
        # Le modèle renvoie directement le token [CLS] normalisé (LayerNorm)
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        preprocess = transforms.Compose([
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),  # 224 = 16 * 14 (patch de 14 px)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        dim = 384

    elif name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()  # on retire le classifieur 1000 classes
        preprocess = weights.transforms()
        dim = 512

    else:
        raise ValueError(f"Unknown model: {name}")

    model = model.to(DEVICE).eval()
    _ENCODERS[name] = (model, preprocess, dim)
    return _ENCODERS[name]


def extract_embeddings(photos, batch_size=16, name=None):
    """
    Retourne une matrice Z de forme (N, d), une ligne L2-normalisée par photo.
    L'ordre des lignes est celui de `photos`.
    """

    photos = list(photos)
    encoder, preprocess, dim = _load_encoder(name)

    if not photos:
        return torch.empty((0, dim), dtype=torch.float32)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    embeddings = []

    with torch.inference_mode():  # pas de gradient
        for start in range(0, len(photos), batch_size):
            batch = []
            for photo in photos[start:start + batch_size]:
                try:
                    with Image.open(photo) as image:
                        batch.append(preprocess(image.convert("RGB")))
                except (OSError, ValueError) as error:
                    raise ValueError(
                        f"Unable to read image: {photo}"
                    ) from error

            x = torch.stack(batch).to(DEVICE)
            features = encoder(x)                       # (B, d)
            features = normalize(features, p=2, dim=1)  # norme = 1
            embeddings.append(features.cpu())

    return torch.cat(embeddings, dim=0)


def _files_signature(photos):
    return [(p.name, p.stat().st_mtime_ns) for p in photos]


def _cache_is_valid(cache, photos, name):
    """Le cache correspond-il au même modèle et aux mêmes fichiers ?"""

    return (
        isinstance(cache, dict)
        and cache.get("model") == name
        and cache.get("files") == _files_signature(photos)
        and isinstance(cache.get("embeddings"), torch.Tensor)
        and cache["embeddings"].ndim == 2
        and cache["embeddings"].shape[0] == len(photos)
    )


def get_embeddings(photo_dir, name=None):
    """Retourne (photos, Z) en utilisant un cache disque propre à chaque modèle."""

    name = name or MODEL_NAME
    photos = get_photos(photo_dir)
    cache_path = Path(photo_dir) / f"{name}_embeddings.pt"

    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if _cache_is_valid(cache, photos, name):
            return photos, cache["embeddings"]

    embeddings = extract_embeddings(photos, name=name)
    torch.save(
        {
            "model": name,
            "files": _files_signature(photos),
            "embeddings": embeddings,
        },
        cache_path,
    )
    return photos, embeddings


def representation_sanity_check(photo_dir, name=None):
    """Vérifications de la section 3.4."""

    photos, Z = get_embeddings(photo_dir, name)
    norms = Z.norm(dim=1)

    return {
        "model": name or MODEL_NAME,
        "shape": tuple(Z.shape),
        "photo_count": len(photos),
        "shape_matches_N": Z.shape[0] == len(photos),
        "dimension": Z.shape[1],
        "norm_min": norms.min().item() if len(photos) else 0.0,
        "norm_max": norms.max().item() if len(photos) else 0.0,
        "has_nan": bool(torch.isnan(Z).any()),
        "unique_vectors": len(torch.unique(Z, dim=0)),
    }


# ---------------------------------------------------------------------------
# Section 4 : Photo Map (PLACEHOLDER -- à remplacer par PCA / t-SNE / UMAP)
# ---------------------------------------------------------------------------

def project_photos(photo_dir):
    """Positions aléatoires déterministes (à remplacer en section 4)."""

    photos = get_photos(photo_dir)
    rng = random.Random(42)

    return [
        {
            "name": photo.name,
            "x": rng.uniform(0.05, 0.95),
            "y": rng.uniform(0.05, 0.95),
        }
        for photo in photos
    ]


# ---------------------------------------------------------------------------
# Section 5 : voisins (PLACEHOLDER -- à remplacer par similarité sur Z)
# ---------------------------------------------------------------------------

def find_neighbours(photo_dir, filename, k=5):
    """Voisins aléatoires déterministes (à remplacer en section 5)."""

    photos = get_photos(photo_dir)
    if not any(photo.name == filename for photo in photos):
        raise ValueError(f"Unknown photo: {filename}")

    candidates = [photo for photo in photos if photo.name != filename]

    rng = random.Random(filename)
    rng.shuffle(candidates)

    return [
        {
            "name": photo.name,
            "similarity": round(rng.uniform(0.60, 0.95), 3),
        }
        for photo in candidates[:k]
    ]


# ---------------------------------------------------------------------------
# Section 6 : groupes (PLACEHOLDER -- à remplacer par un clustering sur Z)
# ---------------------------------------------------------------------------

def discover_groups(photo_dir):
    """Groupes aléatoires sans sens sémantique (à remplacer en section 6)."""

    photos = get_photos(photo_dir)
    rng = random.Random(42)

    n_groups = 5
    groups = [
        {"id": i, "name": f"Group {i + 1}", "photos": []}
        for i in range(n_groups)
    ]

    shuffled_photos = photos.copy()
    rng.shuffle(shuffled_photos)

    for i, photo in enumerate(shuffled_photos):
        groups[i % n_groups]["photos"].append(photo.name)

    return groups


if __name__ == "__main__":
    # Comparaison rapide des deux encodeurs : python pipeline.py
    for m in ("resnet18", "dinov2"):
        print(representation_sanity_check("data/photos", m))