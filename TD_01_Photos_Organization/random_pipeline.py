from pathlib import Path
import random

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SHAPES = [
    "circle",
    "square",
    "triangle",
    "diamond",
]

COLORS = [
    (220, 70, 70),
    (70, 130, 220),
    (70, 180, 100),
    (220, 180, 60),
    (160, 90, 200),
]

BACKGROUNDS = [
    (235, 235, 235),
    (40, 40, 40),
]

def _draw_shape(draw, shape, box, color):

    x0, y0, x1, y1 = box

    if shape == "circle":

        draw.ellipse(
            box,
            fill=color,
        )

    elif shape == "square":

        draw.rectangle(
            box,
            fill=color,
        )

    elif shape == "triangle":

        draw.polygon(
            [
                ((x0 + x1) / 2, y0),
                (x0, y1),
                (x1, y1),
            ],
            fill=color,
        )

    elif shape == "diamond":

        draw.polygon(
            [
                ((x0 + x1) / 2, y0),
                (x1, (y0 + y1) / 2),
                ((x0 + x1) / 2, y1),
                (x0, (y0 + y1) / 2),
            ],
            fill=color,
        )


def generate_mock_photos(
    photo_dir,
    n_photos=100,
    seed=42,
):
    """Generate a synthetic collection of geometric images."""

    photo_dir = Path(photo_dir)
    photo_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = random.Random(seed)

    width = 400
    height = 300

    for i in range(n_photos):

        background = rng.choice(
            BACKGROUNDS
        )

        image = Image.new(
            "RGB",
            (width, height),
            background,
        )

        draw = ImageDraw.Draw(image)

        shape = rng.choice(
            SHAPES
        )

        color = rng.choice(
            COLORS
        )

        size = rng.randint(
            70,
            170,
        )

        center_x = rng.randint(
            100,
            width - 100,
        )

        center_y = rng.randint(
            90,
            height - 90,
        )

        half = size // 2

        box = (
            center_x - half,
            center_y - half,
            center_x + half,
            center_y + half,
        )

        _draw_shape(
            draw,
            shape,
            box,
            color,
        )

        filename = f"IMG_{i:04d}.jpg"

        image.save(
            photo_dir / filename,
            quality=95,
        )

def get_photos(photo_dir):
    """Return all images available in the collection."""

    photo_dir = Path(photo_dir)

    photo_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    photos = [
        path
        for path in sorted(photo_dir.iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Generate the mock collection automatically if necessary.
    if not photos:

        generate_mock_photos(photo_dir)

        photos = [
            path
            for path in sorted(photo_dir.iterdir())
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    return photos

def project_photos(photo_dir):
    """
    Generate random 2D positions for the mock Photo Map.

    This function does not use image representations.
    It only demonstrates the output expected by the frontend.
    """

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


def find_neighbours(photo_dir, filename, k=5):
    """
    Return random photographs as mock neighbours.

    No image similarity is computed.
    """

    photos = get_photos(photo_dir)

    candidates = [
        photo
        for photo in photos
        if photo.name != filename
    ]

    rng = random.Random(filename)

    rng.shuffle(candidates)

    neighbours = candidates[:k]

    return [
        {
            "name": photo.name,
            "similarity": round(
                rng.uniform(0.60, 0.95),
                3,
            ),
        }
        for photo in neighbours
    ]


def discover_groups(photo_dir):
    """
    Randomly assign photographs to mock groups.

    These groups have no semantic meaning and are only used to
    demonstrate the frontend.
    """

    photos = get_photos(photo_dir)

    rng = random.Random(42)

    n_groups = 5

    groups = [
        {
            "id": i,
            "name": f"Group {i + 1}",
            "photos": [],
        }
        for i in range(n_groups)
    ]

    shuffled_photos = photos.copy()
    rng.shuffle(shuffled_photos)

    for i, photo in enumerate(shuffled_photos):
        group_id = i % n_groups

        groups[group_id]["photos"].append(
            photo.name
        )

    return groups