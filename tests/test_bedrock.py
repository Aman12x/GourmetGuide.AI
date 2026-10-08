import pytest

from bedrock import BedrockLLM, GuardrailBlocked, detect_image_format, extract_json
from conftest import FakeConverseClient, tool_use

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


def test_detect_image_format():
    assert detect_image_format(PNG) == "png"
    assert detect_image_format(JPEG) == "jpeg"
    assert detect_image_format(b"GIF89a....") == "gif"
    assert detect_image_format(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert detect_image_format(b"%PDF-1.7") is None


def test_repo_images_are_png_not_jpeg(dishes):
    # The old code labelled every upload image/jpeg; the menu photos themselves are PNG.
    with open("data/" + dishes[0]["image_path"], "rb") as f:
        assert detect_image_format(f.read(16)) == "png"


def test_structured_returns_tool_input_and_forces_the_tool():
    client = FakeConverseClient(lambda kw: tool_use("plan_turn", {"intent": "chat"}))
    llm = BedrockLLM(client)
    out = llm.structured("sys", [{"role": "user", "content": [{"text": "hi"}]}], "plan_turn", "d", {"type": "object"})
    assert out == {"intent": "chat"}
    req = client.requests[0]
    assert req["toolConfig"]["toolChoice"] == {"tool": {"name": "plan_turn"}}
    assert req["additionalModelRequestFields"] == {"thinking": {"type": "disabled"}}
    assert "guardrailConfig" not in req


def test_structured_falls_back_to_json_wrapped_in_prose():
    client = FakeConverseClient(lambda kw: [{"text": 'Sure! Here it is:\n{"intent": "recommend"}\nEnjoy.'}])
    out = BedrockLLM(client).structured("s", [], "plan_turn", "d", {"type": "object"})
    assert out == {"intent": "recommend"}


def test_extract_json_raises_without_object():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_guardrail_config_sent_and_intervention_raised():
    blocked = {
        "output": {"message": {"role": "assistant", "content": [{"text": "I can't help with that."}]}},
        "stopReason": "guardrail_intervened",
    }
    client = FakeConverseClient(lambda kw: blocked)
    llm = BedrockLLM(client, guardrail_id="gr-123", guardrail_version="1")
    with pytest.raises(GuardrailBlocked) as exc:
        llm.text("s", [])
    assert exc.value.message == "I can't help with that."
    assert client.requests[0]["guardrailConfig"] == {"guardrailIdentifier": "gr-123", "guardrailVersion": "1"}
