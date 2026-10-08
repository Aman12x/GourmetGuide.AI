import json
import pickle

from ingest import to_document


def test_embedded_text_names_restaurant_and_cuisine(dishes):
    for dish in dishes:
        doc = to_document(dish)
        assert dish["restaurant_name"] in doc.page_content
        assert dish["cuisine"] in doc.page_content
        assert dish["description"] in doc.page_content


def test_metadata_keeps_the_fields_the_app_reads(dishes):
    m = to_document(dishes[0]).metadata
    for key in ("id", "menu_item_name", "restaurant_name", "cuisine", "price", "vegetarian",
                "dietary_warnings", "ingredients", "average_rating", "calories", "serves", "image_path"):
        assert key in m


def test_dishes_jsonl_matches_the_committed_index(dishes):
    # dishes.jsonl was exported from the committed index, so a rebuild keeps the same 50 descriptions.
    docstore, _ = pickle.load(open("output/faiss_index/index.pkl", "rb"))
    indexed = {d.metadata["id"]: d.page_content for d in docstore._dict.values()}
    assert len(dishes) == len(indexed) == 50
    for dish in dishes:
        assert indexed[dish["id"]].startswith(dish["description"])
