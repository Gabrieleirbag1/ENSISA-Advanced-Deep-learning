from pathlib import Path

from flask import Flask, jsonify, render_template, send_from_directory

from random_pipeline import (
    discover_groups,
    find_neighbours,
    get_photos,
    project_photos,
)


app = Flask(__name__)

PHOTO_DIR = Path("data/mock_photos")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/photos/<path:filename>")
def serve_photo(filename):
    return send_from_directory(PHOTO_DIR.resolve(), filename)


@app.route("/api/photos")
def photos():
    paths = get_photos(PHOTO_DIR)

    return jsonify([
        {
            "name": path.name,
            "url": f"/photos/{path.name}",
        }
        for path in paths
    ])


@app.route("/api/map")
def photo_map():
    photos = project_photos(PHOTO_DIR)

    for photo in photos:
        photo["url"] = f"/photos/{photo['name']}"

    return jsonify(photos)


@app.route("/api/neighbours/<filename>")
def neighbours(filename):

    try:
        photos = find_neighbours(PHOTO_DIR, filename)

    except ValueError:
        return jsonify({"error": "Photo not found"}), 404

    for photo in photos:
        photo["url"] = f"/photos/{photo['name']}"

    return jsonify(photos)


@app.route("/api/groups")
def groups():

    groups = discover_groups(PHOTO_DIR)

    for group in groups:

        group["count"] = len(group["photos"])

        group["photos"] = [
            {
                "name": filename,
                "url": f"/photos/{filename}",
            }
            for filename in group["photos"]
        ]

    return jsonify(groups)


if __name__ == "__main__":
    app.run(debug=True)