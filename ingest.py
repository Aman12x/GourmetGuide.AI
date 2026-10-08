"""Build the FAISS index from data/dishes.jsonl with Amazon Titan Text Embeddings v2.

    python ingest.py                 # writes output/faiss_index
    python ingest.py --out /tmp/idx  # write somewhere else

Needs AWS credentials with Bedrock access to amazon.titan-embed-text-v2:0 in us-east-1.
"""

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document

DISHES = Path("data/dishes.jsonl")


def load_dishes(path=DISHES):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def to_document(dish):
    """Embedded text carries the dish, restaurant and cuisine so name and cuisine queries match."""
    warnings = dish["dietary_warnings"] or "none"
    page_content = (
        f"{dish['menu_item_name']} from {dish['restaurant_name']} ({dish['cuisine']} cuisine). "
        f"{dish['description']} Ingredients: {dish['ingredients']}. {dish['nutrition']}, "
        f"Calories: {dish['calories']}. Dietary warnings: {warnings}. Type: {dish['vegetarian']}. "
        f"Rating: {dish['average_rating']}. Price: ${dish['price']}. Serves: {dish['serves']}."
    )
    metadata = {k: v for k, v in dish.items() if k != "description"}
    metadata["name"] = dish["menu_item_name"]
    return Document(page_content=page_content, metadata=metadata)


def build_index(embeddings, dishes):
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents([to_document(d) for d in dishes], embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/faiss_index")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    import boto3
    from langchain_aws import BedrockEmbeddings

    from bedrock import EMBEDDING_MODEL_ID

    client = boto3.client("bedrock-runtime", region_name=args.region)
    embeddings = BedrockEmbeddings(client=client, model_id=EMBEDDING_MODEL_ID)
    dishes = load_dishes()
    build_index(embeddings, dishes).save_local(args.out)
    print(f"indexed {len(dishes)} dishes into {args.out}")


if __name__ == "__main__":
    main()
