"""Thin wrapper over the Bedrock Converse API.

One client for every model call in the app, so call counting, guardrails and structured
output are handled in one place instead of per prompt.
"""

import json
import re

DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-5-5"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"

_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
)


class GuardrailBlocked(Exception):
    """Raised when a Bedrock Guardrail intervenes on the input or the output."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


def detect_image_format(data: bytes):
    """Return the Converse image format for these bytes, or None if unsupported."""
    for signature, fmt in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return fmt
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def user_message(text, image_bytes=None, image_format=None):
    content = []
    if image_bytes is not None:
        content.append({"image": {"format": image_format, "source": {"bytes": image_bytes}}})
    content.append({"text": text})
    return {"role": "user", "content": content}


def history_messages(history):
    """(user, assistant) text pairs -> alternating Converse messages."""
    messages = []
    for user_text, bot_text in history:
        messages.append({"role": "user", "content": [{"text": user_text or "(image)"}]})
        messages.append({"role": "assistant", "content": [{"text": bot_text or "(no reply)"}]})
    return messages


def extract_json(text):
    """Pull the first JSON object out of free text (models sometimes wrap JSON in prose)."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model output")
    return json.loads(match.group(0))


class BedrockLLM:
    def __init__(
        self,
        client,
        model_id=DEFAULT_MODEL_ID,
        guardrail_id=None,
        guardrail_version="DRAFT",
        max_tokens=1024,
        additional_fields=None,
    ):
        self.client = client
        self.model_id = model_id
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.max_tokens = max_tokens
        # Structured calls force a tool choice, which Claude rejects while extended
        # thinking is on, so thinking is turned off by default.
        self.additional_fields = (
            {"thinking": {"type": "disabled"}} if additional_fields is None else additional_fields
        )
        self.calls = 0

    def _converse(self, system, messages, tool_config=None):
        kwargs = {
            "modelId": self.model_id,
            "system": [{"text": system}],
            "messages": messages,
            "inferenceConfig": {"maxTokens": self.max_tokens},
        }
        if self.additional_fields:
            kwargs["additionalModelRequestFields"] = self.additional_fields
        if tool_config:
            kwargs["toolConfig"] = tool_config
        if self.guardrail_id:
            kwargs["guardrailConfig"] = {
                "guardrailIdentifier": self.guardrail_id,
                "guardrailVersion": self.guardrail_version,
            }
        self.calls += 1
        response = self.client.converse(**kwargs)
        content = response["output"]["message"]["content"]
        if response.get("stopReason") == "guardrail_intervened":
            text = " ".join(block["text"] for block in content if "text" in block)
            raise GuardrailBlocked(text or "Sorry, I can't help with that request.")
        return content

    def text(self, system, messages):
        content = self._converse(system, messages)
        return "".join(block.get("text", "") for block in content).strip()

    def structured(self, system, messages, tool_name, description, schema):
        """Force the model to answer through one tool and return the tool input as a dict."""
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool_name,
                        "description": description,
                        "inputSchema": {"json": schema},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": tool_name}},
        }
        content = self._converse(system, messages, tool_config)
        for block in content:
            if "toolUse" in block and block["toolUse"]["name"] == tool_name:
                return block["toolUse"]["input"]
        text = "".join(block.get("text", "") for block in content)
        return extract_json(text)
