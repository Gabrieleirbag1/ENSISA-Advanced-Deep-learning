#!/usr/bin/env python3
"""
Télécharge des photos de politiciens français depuis Wikidata / Wikimedia Commons.

Usage :
    pip install requests
    python telecharger_politiciens.py                      # 500 photos max
    python telecharger_politiciens.py --limit 3000 --width 600 --out photos

Résultat :
    <out>/Prenom_Nom_Q12345.jpg   (une photo par personne)
    <out>/metadata.csv            (nom, identifiant Wikidata, URL source)

Les photos viennent de Wikimedia Commons : chacune a sa propre licence
(souvent CC BY-SA). Garde metadata.csv pour l'attribution si tu les rediffuses.
"""

import argparse
import csv
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, unquote

import requests

SPARQL_URL = "https://query.wikidata.org/sparql"
# Wikimedia exige un User-Agent identifiable, sinon risque de blocage.
HEADERS = {
    "User-Agent": "PoliticiensFR-PhotoDownloader/1.0 (usage personnel; script Python requests)"
}

QUERY = """
SELECT ?item ?itemLabel ?image WHERE {{
  ?item wdt:P31 wd:Q5;          # être humain
        wdt:P106 wd:Q82955;     # profession : personnalité politique
        wdt:P27 wd:Q142;        # nationalité : France
        wdt:P18 ?image.         # a une image
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "fr,en". }}
}}
LIMIT {limit} OFFSET {offset}
"""


def fetch_people(max_people: int, page_size: int = 500):
    """Récupère (qid, nom, url_image) par pages via SPARQL."""
    people, seen, offset = [], set(), 0
    while len(people) < max_people:
        query = QUERY.format(limit=page_size, offset=offset)
        for attempt in range(4):
            try:
                r = requests.get(
                    SPARQL_URL,
                    params={"query": query, "format": "json"},
                    headers=HEADERS,
                    timeout=90,
                )
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                r.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"  SPARQL erreur ({e}), nouvel essai...")
                time.sleep(5 * (attempt + 1))
        else:
            print("Impossible de joindre Wikidata, arrêt de la collecte.")
            break

        rows = r.json()["results"]["bindings"]
        if not rows:
            break
        for row in rows:
            qid = row["item"]["value"].rsplit("/", 1)[-1]
            if qid in seen:  # une seule photo par personne
                continue
            seen.add(qid)
            name = row["itemLabel"]["value"]
            if re.fullmatch(r"Q\d+", name):  # pas de libellé -> on ignore
                continue
            people.append((qid, name, row["image"]["value"]))
        print(f"  {len(people)} personnes collectées...")
        offset += page_size
        time.sleep(1)
    return people[:max_people]


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "inconnu"


def thumb_url(image_url: str, width: int) -> str:
    """Transforme l'URL Special:FilePath en version redimensionnée en HTTPS."""
    filename = unquote(image_url.rsplit("/", 1)[-1])
    return (
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        f"{quote(filename)}?width={width}"
    )


def download(person, out_dir: Path, width: int):
    qid, name, image_url = person
    base = f"{slugify(name)}_{qid}"
    existing = list(out_dir.glob(base + ".*"))
    if existing:  # reprise : déjà téléchargé
        return qid, name, image_url, existing[0].name, "déjà présent"

    url = thumb_url(image_url, width)
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            if not ctype.startswith("image/"):
                return qid, name, image_url, "", f"pas une image ({ctype})"
            ext = {"image/jpeg": ".jpg", "image/png": ".png",
                   "image/webp": ".webp", "image/gif": ".gif"}.get(ctype.split(";")[0], ".jpg")
            path = out_dir / (base + ext)
            path.write_bytes(r.content)
            time.sleep(0.3)  # politesse envers Commons
            return qid, name, image_url, path.name, "ok"
        except requests.RequestException as e:
            err = str(e)
            time.sleep(2 * (attempt + 1))
    return qid, name, image_url, "", f"échec : {err}"


def main():
    ap = argparse.ArgumentParser(description="Photos de politiciens français (Wikidata/Commons)")
    ap.add_argument("--limit", type=int, default=500, help="nombre max de personnes (défaut 500)")
    ap.add_argument("--width", type=int, default=800, help="largeur des images en px (défaut 800)")
    ap.add_argument("--out", default="politiciens_fr", help="dossier de sortie")
    ap.add_argument("--workers", type=int, default=3, help="téléchargements parallèles (garde un petit nombre)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Requête Wikidata...")
    people = fetch_people(args.limit)
    print(f"{len(people)} personnes avec photo. Téléchargement vers '{out_dir}/'...")

    results, ok = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, p, out_dir, args.width) for p in people]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if res[4] in ("ok", "déjà présent"):
                ok += 1
            if i % 25 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} traités ({ok} réussis)")

    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["wikidata_id", "nom", "source_commons", "fichier", "statut"])
        w.writerows(sorted(results, key=lambda r: r[1]))

    print(f"Terminé : {ok} photos dans '{out_dir}/' (détails dans metadata.csv).")


if __name__ == "__main__":
    main()