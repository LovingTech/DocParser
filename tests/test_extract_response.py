"""Tests for model-response recovery, schema coercion, and the retry loop."""

import asyncio

import pytest

from backend.config import Settings
from backend.models import FieldSchema, FieldType, Schema
from backend.services import Document
from backend.services.document_parser_service import DocumentParserService
from backend.services.extract_response import (
    ExtractResponseError,
    coerce_value,
    normalize_response,
    parse_extracted,
    repair_json,
)


SCHEMA = Schema(
    name="invoice",
    fields=[
        FieldSchema(name="invoice_number", type=FieldType.STRING, description="Number"),
        FieldSchema(name="total", type=FieldType.NUMBER, description="Total"),
        FieldSchema(name="qty", type=FieldType.INTEGER, description="Qty"),
        FieldSchema(name="paid", type=FieldType.BOOLEAN, description="Paid"),
    ],
)


class RecordingLLM:
    """Fake LLM that returns/raises whatever its side effect specifies."""

    def __init__(self, side_effect):
        self.calls = 0
        self._side_effect = side_effect

    def chat_completion(self, messages):
        self.calls += 1
        return self._side_effect(messages)


class FakePdfService:
    def convert(self, doc):
        return []


def make_service(llm, **settings_kwargs):
    settings = Settings()
    for key, value in settings_kwargs.items():
        setattr(settings, key, value)
    return DocumentParserService(
        llm=llm,
        pdf_to_image_conversion_service=FakePdfService(),
        settings=settings,
    )


async def _parse(service, schema=None):
    doc = Document(pdf_bytes=b"%PDF-1.4 test", schema=schema or SCHEMA)
    return await service.parse(doc)


# --------------------------------------------------------------------------- #
# repair_json
# --------------------------------------------------------------------------- #
def test_repair_parses_clean_json():
    assert repair_json('{"a": 1}') == '{"a": 1}'


def test_repair_strips_code_fence():
    content = "```json\n{\"a\": 2}\n```"
    assert repair_json(content) == '{"a": 2}'


def test_repair_skips_leading_prose_and_trailing_text():
    content = "Sure! Here you go: {\"a\": 1} hope this helps!"
    assert repair_json(content) == '{"a": 1}'


def test_repair_removes_trailing_commas():
    assert repair_json('{"a": 1, "b": 2,}') == '{"a": 1, "b": 2}'


def test_repair_converts_single_quotes():
    content = "{'a': 'x', 'b': 3}"
    assert repair_json(content) == '{"a": "x", "b": 3}'


def test_repair_replaces_js_ism():
    assert repair_json('{"a": undefined}') == '{"a": null}'


def test_repair_returns_none_when_no_object():
    assert repair_json("there is no json in this reply") is None


# --------------------------------------------------------------------------- #
# parse_extracted / normalize_response / coerce_value
# --------------------------------------------------------------------------- #
def test_parse_extracted_returns_dict_for_object():
    result = parse_extracted('{"invoice_number": "INV-1"}', SCHEMA)
    assert result == {"invoice_number": "INV-1"}


def test_parse_extracted_raises_for_non_object():
    with pytest.raises(ExtractResponseError):
        parse_extracted("[1, 2, 3]", SCHEMA)


def test_parse_extracted_raises_for_garbage():
    with pytest.raises(ExtractResponseError):
        parse_extracted("completely unparseable text", SCHEMA)


def test_normalize_keeps_only_schema_fields():
    data = {"invoice_number": "INV-9", "unexpected": 123}
    result = normalize_response(data, SCHEMA)
    assert result == {"invoice_number": "INV-9"}
    assert "unexpected" not in result


def test_normalize_drops_missing_fields():
    assert normalize_response({}, SCHEMA) == {}


def test_coerce_number_with_commas_and_currency():
    field = FieldSchema(name="total", type=FieldType.NUMBER)
    assert coerce_value("$1,250.00", field) == 1250.0


def test_coerce_integer_from_string():
    field = FieldSchema(name="qty", type=FieldType.INTEGER)
    assert coerce_value("42", field) == 42


def test_coerce_boolean_from_string():
    for text, expected in [("true", True), ("yes", True), ("0", False), ("no", False)]:
        assert coerce_value(text, FieldSchema(name="paid", type=FieldType.BOOLEAN)) is expected


def test_coerce_any_is_identity():
    field = FieldSchema(name="anything", type=FieldType.ANY)
    assert coerce_value(["a", "b"], field) == ["a", "b"]


def test_coerce_single_item_array_wrapped():
    field = FieldSchema(
        name="items",
        type=FieldType.ARRAY,
        items=[FieldSchema(name="qty", type=FieldType.INTEGER)],
    )
    assert coerce_value({"qty": "5"}, field) == [{"qty": 5}]


def test_coerce_array_of_objects():
    field = FieldSchema(
        name="items",
        type=FieldType.ARRAY,
        items=[FieldSchema(name="qty", type=FieldType.INTEGER)],
    )
    result = coerce_value([{"qty": "1"}, {"qty": "2"}], field)
    assert result == [{"qty": 1}, {"qty": 2}]


def test_coerce_object():
    field = FieldSchema(
        name="address",
        type=FieldType.OBJECT,
        items=[FieldSchema(name="city", type=FieldType.STRING)],
    )
    result = coerce_value({"city": "NYC", "junk": 1}, field)
    assert result == {"city": "NYC"}


# --------------------------------------------------------------------------- #
# retry loop behaviour
# --------------------------------------------------------------------------- #
def test_parse_recovers_malformed_response_without_retry():
    llm = RecordingLLM(lambda m: "Sure! Here is the JSON: {\"invoice_number\": \"X\", \"total\": 5}")
    service = make_service(llm)
    result = asyncio.run(_parse(service))
    assert result == {"invoice_number": "X", "total": 5.0}
    assert llm.calls == 1  # recoverable -> no retry


def test_parse_retries_then_succeeds_on_bad_content():
    bad = "I cannot extract anything useful."
    good = '{"invoice_number": "OK"}'
    states = {"n": 0}

    def side_effect(m):
        states["n"] += 1
        return bad if states["n"] == 1 else good

    llm = RecordingLLM(side_effect)
    service = make_service(llm, llm_retries=2)
    result = asyncio.run(_parse(service))
    assert result == {"invoice_number": "OK"}
    assert llm.calls == 2  # one retry


def test_parse_raises_after_exhausting_retries_on_bad_content():
    llm = RecordingLLM(lambda m: "garbage with no json")
    service = make_service(llm, llm_retries=2)
    with pytest.raises(ExtractResponseError):
        asyncio.run(_parse(service))
    assert llm.calls == 3  # initial + 2 retries


def test_parse_retries_then_succeeds_on_transient_error():
    good = '{"invoice_number": "again"}'
    states = {"n": 0}

    def side_effect(m):
        states["n"] += 1
        if states["n"] == 1:
            raise RuntimeError("network hiccup")
        return good

    llm = RecordingLLM(side_effect)
    service = make_service(llm, llm_retries=2)
    result = asyncio.run(_parse(service))
    assert result == {"invoice_number": "again"}
    assert llm.calls == 2


def test_parse_raises_after_exhausting_retries_on_transient_error():
    def boom(m):
        raise RuntimeError("network hiccup")

    llm = RecordingLLM(boom)
    service = make_service(llm, llm_retries=2)
    with pytest.raises(RuntimeError):
        asyncio.run(_parse(service))
    assert llm.calls == 3


def test_backoff_uses_asyncio_sleep_non_blocking():
    bad = "garbage with no json"
    good = '{"invoice_number": "ok"}'
    states = {"n": 0}

    def side_effect(m):
        states["n"] += 1
        return bad if states["n"] == 1 else good

    llm = RecordingLLM(side_effect)
    service = make_service(llm, llm_retries=1, llm_retry_backoff=0.5)
    import unittest.mock

    with unittest.mock.patch(
        "backend.services.document_parser_service.asyncio.sleep"
    ) as slept:
        asyncio.run(_parse(service))
    slept.assert_awaited_once_with(0.5)
