import pytest

from bedrock import BedrockLLM
from conftest import FakeConverseClient, tool_name, tool_use
from pipeline import UnsupportedImage, handle_turn

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def scripted(plan, picks=None):
    """Answer the plan call with `plan`; the rank call with `picks` (default: first two candidates)."""

    def respond(kw):
        name = tool_name(kw)
        if name == "plan_turn":
            return tool_use(name, plan)
        if name == "recommend_dishes":
            text = kw["messages"][0]["content"][-1]["text"]
            ids = [line[4:] for line in text.splitlines() if line.startswith("id: ")]
            chosen = picks if picks is not None else ids[:2]
            return tool_use(name, {"intro": "Try these:", "picks": [{"id": i, "summary": f"why {i}"} for i in chosen]})
        return [{"text": "a bowl of ramen with egg and pork broth"}]  # image description

    return respond


def test_recommendation_turn_uses_two_model_calls(db, cuisines):
    plan = {"intent": "recommend", "search_query": "spicy vegetarian", "vegetarian": True, "max_price": 10}
    client = FakeConverseClient(scripted(plan))
    llm = BedrockLLM(client)
    turn = handle_turn(llm, db, cuisines, "something spicy and vegetarian under $10")
    assert llm.calls == 2
    assert len(turn.recommendations) == 2
    for metadata, summary in turn.recommendations:
        assert metadata["vegetarian"] == "Vegetarian" and metadata["price"] <= 10
        assert summary == f"why {metadata['id']}"


def test_image_turn_adds_exactly_one_call_and_sends_real_format(db, cuisines):
    client = FakeConverseClient(scripted({"intent": "recommend", "search_query": "ramen"}))
    llm = BedrockLLM(client)
    handle_turn(llm, db, cuisines, "", PNG)
    assert llm.calls == 3
    image_block = client.requests[0]["messages"][0]["content"][0]["image"]
    assert image_block["format"] == "png"
    assert "ramen" in client.requests[1]["messages"][-1]["content"][-1]["text"]


def test_chat_turn_is_one_call_and_skips_retrieval(db, cuisines):
    client = FakeConverseClient(scripted({"intent": "chat", "reply": "Hi! What are you craving?"}))
    llm = BedrockLLM(client)
    turn = handle_turn(llm, db, cuisines, "hello")
    assert llm.calls == 1
    assert turn.text == "Hi! What are you craving?" and turn.recommendations == []


def test_no_match_answers_without_calling_the_ranker(db, cuisines):
    client = FakeConverseClient(scripted({"intent": "recommend", "search_query": "x", "max_price": 1}))
    llm = BedrockLLM(client)
    turn = handle_turn(llm, db, cuisines, "dinner under a dollar")
    assert llm.calls == 1
    assert "up to $1" in turn.text and turn.recommendations == []


def test_invented_and_duplicate_ids_are_dropped(db, cuisines, dishes):
    real = next(d["id"] for d in dishes if d["cuisine"] == "Italian")
    plan = {"intent": "recommend", "search_query": "pasta", "cuisines": ["Italian"]}
    client = FakeConverseClient(scripted(plan, picks=["FAKE01", real, real]))
    turn = handle_turn(BedrockLLM(client), db, cuisines, "pasta")
    assert [m["id"] for m, _ in turn.recommendations] == [real]


def test_history_is_sent_to_the_planner(db, cuisines):
    client = FakeConverseClient(scripted({"intent": "chat", "reply": "ok"}))
    history = [("show me pasta", "Here are some pastas")]
    handle_turn(BedrockLLM(client), db, cuisines, "something cheaper?", history=history)
    texts = [m["content"][0]["text"] for m in client.requests[0]["messages"]]
    assert texts == ["show me pasta", "Here are some pastas", "something cheaper?"]


def test_unsupported_upload_is_rejected_before_any_call(db, cuisines):
    client = FakeConverseClient(scripted({"intent": "chat"}))
    llm = BedrockLLM(client)
    with pytest.raises(UnsupportedImage):
        handle_turn(llm, db, cuisines, "", b"%PDF-1.7 not an image")
    assert llm.calls == 0
