"""Validate the guardrail request against botocore's Bedrock service model (offline, no AWS call)."""

import sys
from pathlib import Path

import botocore.session
from botocore.validate import ParamValidator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from create_guardrail import GUARDRAIL  # noqa: E402


def _input_shape(service, operation):
    model = botocore.session.get_session().get_service_model(service)
    return model.operation_model(operation).input_shape


def test_create_guardrail_request_matches_the_api_shape():
    report = ParamValidator().validate(GUARDRAIL, _input_shape("bedrock", "CreateGuardrail"))
    assert not report.has_errors(), report.generate_report()


def test_converse_request_with_guardrail_and_tools_matches_the_api_shape():
    from bedrock import BedrockLLM
    from conftest import FakeConverseClient, tool_use

    client = FakeConverseClient(lambda kw: tool_use("plan_turn", {}))
    llm = BedrockLLM(client, guardrail_id="gr-1", guardrail_version="1")
    llm.structured("sys", [{"role": "user", "content": [{"text": "hi"}]}], "plan_turn", "d", {"type": "object"})
    report = ParamValidator().validate(client.requests[0], _input_shape("bedrock-runtime", "Converse"))
    assert not report.has_errors(), report.generate_report()


def test_converse_image_request_matches_the_api_shape():
    from bedrock import BedrockLLM
    from conftest import FakeConverseClient
    from pipeline import describe_image

    client = FakeConverseClient(lambda kw: [{"text": "ramen"}])
    describe_image(BedrockLLM(client), b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    report = ParamValidator().validate(client.requests[0], _input_shape("bedrock-runtime", "Converse"))
    assert not report.has_errors(), report.generate_report()
