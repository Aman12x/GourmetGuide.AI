"""One-time export of the dishes the live index serves into data/dishes.jsonl.

The committed FAISS index was built from an earlier image-description run than the one in
data/menu_descriptions_data.csv (only 21 of 50 descriptions match). This script recovers the
exact descriptions and metadata the index holds, so ingest.py can rebuild it reproducibly.
"""

import json
import pickle
from pathlib import Path

INDEX_PKL = Path("output/faiss_index/index.pkl")
OUT = Path("data/dishes.jsonl")
SEP = ", Ingredients: "


def main():
    docstore, index_to_id = pickle.load(INDEX_PKL.open("rb"))
    rows = []
    for pos in sorted(index_to_id):
        doc = docstore.search(index_to_id[pos])
        description, _ = doc.page_content.split(SEP, 1)
        m = doc.metadata
        rows.append(
            {
                "id": m["id"],
                "menu_item_name": m["menu_item_name"],
                "restaurant_name": m["restaurant_name"],
                "cuisine": m["cuisine"],
                "ingredients": m["ingredients"],
                "nutrition": m["nutrition"].strip(),
                "calories": m["calories"],
                "dietary_warnings": m["dietary_warnings"].strip(),
                "vegetarian": m["vegetarian"],
                "average_rating": m["average_rating"],
                "price": m["price"],
                "serves": m["serves"].strip('"'),
                "image_path": m["image_path"],
                "description": description.strip(),
            }
        )
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} dishes to {OUT}")


if __name__ == "__main__":
    main()
