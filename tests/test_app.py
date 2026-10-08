"""Boot the real Streamlit script against the committed index with Bedrock faked out."""

from unittest import mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import FakeConverseClient, HashEmbeddings, tool_name, tool_use


class Titan1024(HashEmbeddings):
    dim = 1024  # the committed index holds 1024-dim Titan v2 vectors

    def __init__(self, *args, **kwargs):
        pass


def respond(kw):
    if tool_name(kw) == "plan_turn":
        return tool_use("plan_turn", {"intent": "recommend", "search_query": "pizza", "cuisines": ["Pizza"], "max_price": 12})
    text = kw["messages"][0]["content"][-1]["text"]
    first = next(line[4:] for line in text.splitlines() if line.startswith("id: "))
    return tool_use("recommend_dishes", {"intro": "Pizza time:", "picks": [{"id": first, "summary": "Cheesy and cheap."}]})


@pytest.fixture
def app(monkeypatch):
    monkeypatch.chdir(__import__("pathlib").Path(__file__).resolve().parents[1])
    st.cache_resource.clear()  # load_resources is cached across AppTest runs in one process
    client = FakeConverseClient(respond)
    with mock.patch("boto3.client", return_value=client), mock.patch("langchain_aws.BedrockEmbeddings", Titan1024):
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        yield at, client


def test_app_boots_on_committed_index(app):
    at, _ = app
    assert not at.exception
    assert not at.error
    assert "GourmetGuide AI" in [t.value for t in at.title]


def test_text_turn_renders_a_filtered_recommendation_card(app):
    at, client = app
    at.text_input(key="input_field").set_value("pizza under $12")
    next(b for b in at.button if b.label.startswith("Send")).click()
    at.run()
    assert not at.exception
    assert len(client.requests) == 2
    assert any("Pizza time:" in m.value for m in at.markdown)
    assert any("Cheesy and cheap." in m.value for m in at.markdown)
    prices = [m.value for m in at.metric if m.label == "Price"]
    assert len(prices) == 1 and int(prices[0].lstrip("$")) <= 12
