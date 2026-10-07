"""
PhotoMap pipeline.

Section 3 (représentations) : implémentée (DINOv2 ou ResNet-18).
Section 4 (Photo Map) : implémentée (UMAP / t-SNE / PCA).
Section 5 (voisins) : similarité cosinus sur Z.
Section 6 (groupes) : K-Means sur Z, K choisi par silhouette.
Section 7 : fonctions d'évaluation (compare_*).
"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import normalize
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# "dinov2" (auto-supervisé, ViT-S/14, d=384) ou "resnet18" (supervisé, d=512)
# MODEL_NAME = "dinov2"
MODEL_NAME = "resnet18"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_ENCODERS = {}  # cache mémoire : nom -> (encodeur, preprocess, dimension)

# "umap", "tsne" ou "pca"
PROJECTION_METHOD = "umap"
_PROJECTION_CACHE = {}  # (modèle, méthode, fichiers) -> liste de positions


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
# Section 4 : Photo Map (réduction de dimension d -> 2)
# ---------------------------------------------------------------------------

def reduce_to_2d(Z, method=None, random_state=42):
    """
    Projette Z (N, d) vers Y (N, 2) avec PCA, t-SNE ou UMAP.

    Z est supposé L2-normalisé : la métrique "cosine" est donc cohérente
    avec la similarité utilisée pour la recherche de voisins.
    """

    method = method or PROJECTION_METHOD
    Z = np.asarray(Z, dtype=np.float32)
    n = len(Z)

    # Cas dégénérés : trop peu de photos pour t-SNE / UMAP
    if n == 0:
        return np.empty((0, 2), dtype=np.float32)
    if n == 1:
        return np.zeros((1, 2), dtype=np.float32)
    if n < 5:
        method = "pca"

    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=random_state).fit_transform(Z)

    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(
            n_components=2,
            perplexity=min(30.0, (n - 1) / 3),  # doit rester < N
            metric="cosine",
            init="pca",              # initialisation stable et déterministe
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(Z)

    if method == "umap":
        try:
            import umap
        except ImportError:
            print("umap-learn n'est pas installé (pip install umap-learn) : "
                  "repli sur t-SNE.")
            return reduce_to_2d(Z, "tsne", random_state)

        return umap.UMAP(
            n_components=2,
            n_neighbors=min(15, n - 1),
            min_dist=0.1,
            metric="cosine",
            random_state=random_state,
        ).fit_transform(Z)

    raise ValueError(f"Unknown projection method: {method}")


def normalize_coordinates(Y, margin=0.05):
    """
    Min-max indépendant sur chaque axe vers [0, 1], puis rétrécissement
    vers [margin, 1 - margin] pour que les vignettes ne soient pas coupées
    par le bord de la carte.
    """

    Y = np.asarray(Y, dtype=np.float64)
    if len(Y) == 0:
        return Y

    mins = Y.min(axis=0)
    span = Y.max(axis=0) - mins
    constant = span == 0
    span[constant] = 1.0

    unit = (Y - mins) / span
    unit[:, constant] = 0.5  # axe constant : on centre

    return margin + unit * (1.0 - 2.0 * margin)


def project_photos(photo_dir, method=None):
    """
    Retourne [{"name", "x", "y"}, ...] dans l'ordre de get_photos().
    Le résultat est mis en cache : t-SNE / UMAP sont trop lents pour être
    recalculés à chaque appel de /api/map.
    """

    method = method or PROJECTION_METHOD
    photos, Z = get_embeddings(photo_dir)

    key = (MODEL_NAME, method, tuple(_files_signature(photos)))
    if key in _PROJECTION_CACHE:
        return [dict(item) for item in _PROJECTION_CACHE[key]]

    Y = normalize_coordinates(reduce_to_2d(Z.numpy(), method))

    result = [
        {"name": photo.name, "x": float(u), "y": float(v)}
        for photo, (u, v) in zip(photos, Y)
    ]
    _PROJECTION_CACHE[key] = result
    return [dict(item) for item in result]


def compare_projections(photo_dir, methods=("pca", "tsne", "umap"), k=10):
    """
    Compare les méthodes sans labels : la "trustworthiness" mesure à quel
    point les k plus proches voisins dans la carte 2D sont aussi proches
    dans l'espace d'origine (1 = parfait).
    """

    import time
    from sklearn.manifold import trustworthiness

    photos, Z = get_embeddings(photo_dir)
    Z = Z.numpy()
    k = min(k, max(1, len(Z) // 2 - 1))

    report = {}
    for method in methods:
        start = time.time()
        Y = reduce_to_2d(Z, method)
        report[method] = {
            "trustworthiness": float(trustworthiness(Z, Y, n_neighbors=k)),
            "seconds": round(time.time() - start, 2),
        }
    return report


# ---------------------------------------------------------------------------
# Section 5 : recherche de photos similaires (dans l'espace d'origine Z)
# ---------------------------------------------------------------------------

def find_neighbours(photo_dir, filename, k=5):
    """
    Retourne les k photos les plus proches de `filename` selon la similarité
    cosinus calculée sur Z (et non sur la carte 2D, qui déforme les distances).

    Les vecteurs étant L2-normalisés, cosine(zi, zj) = zi . zj : un seul
    produit matrice-vecteur donne la similarité avec toute la collection.
    Le score affiché est directement ce cosinus (dans [-1, 1], proche de 1
    = très similaire). La photo requête est exclue des résultats.
    """

    photos, Z = get_embeddings(photo_dir)

    names = [photo.name for photo in photos]
    if filename not in names:
        raise ValueError(f"Unknown photo: {filename}")

    query = names.index(filename)
    similarities = Z @ Z[query]            # (N,)
    similarities[query] = float("-inf")    # on s'exclut soi-même

    k = max(0, min(k, len(photos) - 1))
    values, indices = torch.topk(similarities, k)

    return [
        {"name": names[i], "similarity": round(float(v), 3)}
        for v, i in zip(values.tolist(), indices.tolist())
    ]


# ---------------------------------------------------------------------------
# Section 6 : découverte automatique de groupes
# ---------------------------------------------------------------------------

CLUSTER_METHOD = "kmeans"   # "kmeans", "agglomerative" ou "hdbscan"
K_MIN = 3
K_MAX = 25
_GROUPS_CACHE = {}


def _silhouette(Z, labels):
    """Silhouette (cosinus) en ignorant le bruit (-1) ; None si non défini."""

    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = len(set(labels[mask].tolist()))
    if n_clusters < 2 or n_clusters > mask.sum() - 1:
        return None
    return float(silhouette_score(Z[mask], labels[mask], metric="cosine"))


def _fit_labels(X, method, k=None):
    """Applique l'algorithme choisi et retourne un label par ligne de X."""

    if method == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)

    if method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(X)

    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN

        # Sur des vecteurs normalisés, euclidien == cosinus (même classement)
        return HDBSCAN(min_cluster_size=max(5, len(X) // 50)).fit_predict(X)

    raise ValueError(f"Unknown clustering method: {method}")


def select_k_by_silhouette(Z, method="kmeans", k_min=K_MIN, k_max=K_MAX):
    """
    Teste plusieurs K et garde celui qui maximise la silhouette.
    Retourne (meilleur_k, {k: score}).
    """

    k_max = min(k_max, len(Z) - 1)
    scores = {}
    for k in range(k_min, k_max + 1):
        score = _silhouette(Z, _fit_labels(Z, method, k))
        if score is not None:
            scores[k] = score

    if not scores:
        return None, scores
    return max(scores, key=scores.get), scores


def cluster_embeddings(Z, method=None, k=None):
    """Retourne (labels, k_utilisé). K est choisi automatiquement si absent."""

    method = method or CLUSTER_METHOD
    Z = np.asarray(Z, dtype=np.float32)

    if len(Z) < 4:
        return np.zeros(len(Z), dtype=int), 1

    if method == "hdbscan":
        labels = _fit_labels(Z, method)
        return labels, len(set(labels.tolist()) - {-1})

    if k is None:
        k, _ = select_k_by_silhouette(Z, method)
        if k is None:
            return np.zeros(len(Z), dtype=int), 1

    return _fit_labels(Z, method, k), k


def discover_groups(photo_dir, method=None):
    """
    Regroupe les photos par clustering NON supervisé de Z (espace d'origine).
    Format attendu par app.py : [{"id", "name", "photos": [noms]}].
    Groupes triés par taille décroissante ; le bruit éventuel de HDBSCAN est
    rassemblé dans un dernier groupe "Unassigned".
    """

    method = method or CLUSTER_METHOD
    photos, Z = get_embeddings(photo_dir)

    key = (MODEL_NAME, method, tuple(_files_signature(photos)))
    if key not in _GROUPS_CACHE:
        labels, _ = cluster_embeddings(Z.numpy(), method)

        members = {}
        for photo, label in zip(photos, labels.tolist()):
            members.setdefault(label, []).append(photo.name)

        ordered = sorted(
            (lab for lab in members if lab != -1),
            key=lambda lab: len(members[lab]),
            reverse=True,
        )
        groups = [
            {"id": i, "name": f"Group {i + 1}", "photos": members[lab]}
            for i, lab in enumerate(ordered)
        ]
        if -1 in members:
            groups.append({
                "id": len(groups),
                "name": "Unassigned",
                "photos": members[-1],
            })
        _GROUPS_CACHE[key] = groups

    return [
        {**group, "photos": list(group["photos"])}
        for group in _GROUPS_CACHE[key]
    ]


# ---------------------------------------------------------------------------
# Section 7 : évaluation (sans labels)
# ---------------------------------------------------------------------------

def compare_encoders(photo_dir, encoders=("resnet18", "dinov2")):
    """
    Compare deux encodeurs : silhouette (cosinus) du meilleur K-Means.
    Attention : la silhouette est calculée dans l'espace propre à chaque
    encodeur ; à compléter par une inspection qualitative des voisins.
    """

    report = {}
    for name in encoders:
        _, Z = get_embeddings(photo_dir, name)
        Z = Z.numpy()
        k, scores = select_k_by_silhouette(Z)
        report[name] = {
            "best_k": k,
            "silhouette": scores.get(k),
            "scores_by_k": {kk: round(s, 4) for kk, s in scores.items()},
        }
    return report


def compare_cluster_spaces(photo_dir, k=None):
    """
    Cluster Z (d dimensions) puis la carte Y (2D) avec le même K, et mesure
    les deux partitions dans l'espace d'origine Z, plus la proportion de
    voisins communs entre Z et Y.
    """

    from sklearn.neighbors import NearestNeighbors

    photos, Z = get_embeddings(photo_dir)
    Z = Z.numpy()
    Y = reduce_to_2d(Z)

    if k is None:
        k, _ = select_k_by_silhouette(Z)
    labels_z = _fit_labels(Z, "kmeans", k)
    labels_y = _fit_labels(Y, "kmeans", k)

    n_nb = min(10, len(Z) - 1)
    nn_z = NearestNeighbors(n_neighbors=n_nb + 1, metric="cosine").fit(Z)
    nn_y = NearestNeighbors(n_neighbors=n_nb + 1).fit(Y)
    idx_z = nn_z.kneighbors(Z, return_distance=False)[:, 1:]
    idx_y = nn_y.kneighbors(Y, return_distance=False)[:, 1:]
    overlap = float(np.mean([
        len(set(a) & set(b)) / n_nb for a, b in zip(idx_z, idx_y)
    ]))

    return {
        "k": k,
        "silhouette_clustering_on_Z": _silhouette(Z, labels_z),
        "silhouette_clustering_on_Y": _silhouette(Z, labels_y),
        "neighbour_overlap_Z_vs_Y": overlap,
    }


if __name__ == "__main__":
    # Comparaison rapide des deux encodeurs : python pipeline.py
    for m in ("resnet18", "dinov2"):
        print(representation_sanity_check("data/photos", m))
    print(compare_projections("data/photos"))
    print(compare_encoders("data/photos"))
    print(compare_cluster_spaces("data/photos"))