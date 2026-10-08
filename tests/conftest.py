import hashlib
import re
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest import build_index, load_dishes  # noqa: E402


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings so tests run offline without Bedrock."""

    dim = 256

    def _embed(self, text):
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        return vec

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class FakeConverseClient:
    """Stands in for boto3's bedrock-runtime client. `responder(kwargs)` returns the Converse content."""

    def __init__(self, responder):
        self.responder = responder
        self.requests = []

    def converse(self, **kwargs):
        self.requests.append(kwargs)
        result = self.responder(kwargs)
        if isinstance(result, dict) and "stopReason" in result:
            return result
        return {"output": {"message": {"role": "assistant", "content": result}}, "stopReason": "end_turn"}


def tool_name(kwargs):
    tool_config = kwargs.get("toolConfig")
    return tool_config["tools"][0]["toolSpec"]["name"] if tool_config else None


def tool_use(name, payload):
    return [{"toolUse": {"toolUseId": "t1", "name": name, "input": payload}}]


@pytest.fixture(scope="session")
def dishes():
    return load_dishes(ROOT / "data/dishes.jsonl")


@pytest.fixture(scope="session")
def db(dishes):
    return build_index(HashEmbeddings(), dishes)


@pytest.fixture(scope="session")
def cuisines(dishes):
    return sorted({d["cuisine"] for d in dishes})
