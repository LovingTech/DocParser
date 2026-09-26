"""Tests for schema handling, including array-of-objects fields."""

import pytest

from backend.models import FieldSchema, FieldType, Schema
from backend.services.document_parser_service import _format_fields


def test_scalar_field_defaults():
    fs = FieldSchema(name="vendor")
    assert fs.type == FieldType.STRING
    assert fs.items == []


def test_array_field_roundtrip_with_items():
    raw = {"name": "line_items", "type": "array", "items": [{"name": "qty", "type": "integer"}]}
    fs = FieldSchema(**raw)
    assert fs.type == FieldType.ARRAY
    assert len(fs.items) == 1
    assert fs.items[0].name == "qty"
    assert fs.items[0].type == FieldType.INTEGER


def test_object_field_can_carry_items():
    fs = FieldSchema(name="billing_address", type=FieldType.OBJECT, items=[
        FieldSchema(name="city", type=FieldType.STRING, description="City"),
    ])
    assert fs.type == FieldType.OBJECT
    assert fs.items[0].name == "city"


def test_invalid_type_rejected():
    with pytest.raises(ValueError):
        FieldSchema(name="x", type="not-a-type")


def test_format_fields_includes_array_items():
    schema = Schema(name="invoice", fields=[
        FieldSchema(name="line_items", type=FieldType.ARRAY, description="Each item row",
                    items=[
                        FieldSchema(name="qty", type=FieldType.INTEGER, description="Quantity"),
                        FieldSchema(name="amount", type=FieldType.NUMBER, description="Line total"),
                    ]),
    ])
    text = _format_fields(schema.fields)
    assert "line_items (array)" in text
    assert "qty (integer)" in text
    assert "amount (number)" in text
    assert "array of objects" in text


def test_format_fields_without_items_has_no_item_block():
    schema = Schema(name="invoice", fields=[
        FieldSchema(name="vendor", type=FieldType.STRING, description="Vendor name"),
    ])
    text = _format_fields(schema.fields)
    assert "vendor (string)" in text
    assert "array of objects" not in text
