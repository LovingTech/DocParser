"""End-to-end tests for the document parser service.

Run with:  uv run pytest -v
The VLM call is mocked, so no API key is required.
"""

import base64
import json
from unittest.mock import AsyncMock, patch

import pytest
import pymupdf
from fastapi.testclient import TestClient

from backend.main import app

SCHEMA = {
    "name": "invoice",
    "fields": [
        {"name": "invoice_number", "type": "string", "description": "Invoice number"},
        {"name": "date", "type": "date", "description": "Invoice issue date"},
        {"name": "total", "type": "number", "description": "Total amount due"},
        {"name": "vendor", "type": "string", "description": "Vendor name"},
    ],
}

FAKE_DATA = {
    "invoice_number": "INV-2024-0042",
    "date": "2024-03-15",
    "total": 1250.00,
    "vendor": "Acme Office Supplies Ltd",
}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_pdf(tmp_path) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        (
            "INVOICE\n"
            "Invoice Number: INV-2024-0042\n"
            "Date: 2024-03-15\n"
            "Vendor: Acme Office Supplies Ltd\n"
            "Total amount due: $1,250.00\n"
        ),
        fontsize=14,
    )
    path = tmp_path / "invoice.pdf"
    doc.save(str(path))
    doc.close()
    return path.read_bytes()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] == "gpt-4o"


def test_extract_multipart(client, sample_pdf):
    with patch("backend.main.parser_service.parse", new=AsyncMock(return_value=FAKE_DATA)) as mocked:
        resp = client.post(
            "/extract",
            files={"file": ("invoice.pdf", sample_pdf, "application/pdf")},
            data={"schema": json.dumps(SCHEMA)},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == FAKE_DATA
    assert body["schema_name"] == "invoice"
    assert body["request_id"]
    assert mocked.called


def test_extract_json_body(client, sample_pdf):
    with patch("backend.main.parser_service.parse", new=AsyncMock(return_value=FAKE_DATA)) as mocked:
        resp = client.post(
            "/extract/json",
            json={
                "file_base64": base64.b64encode(sample_pdf).decode(),
                "schema": SCHEMA,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == FAKE_DATA
    assert mocked.called


def test_extract_rejects_non_pdf(client):
    resp = client.post(
        "/extract",
        files={"file": ("note.txt", b"hello", "text/plain")},
        data={"schema": json.dumps(SCHEMA)},
    )
    assert resp.status_code == 400
    assert "not a valid PDF" in resp.json()["detail"]


def test_extract_rejects_invalid_schema_json(client, sample_pdf):
    resp = client.post(
        "/extract",
        files={"file": ("invoice.pdf", sample_pdf, "application/pdf")},
        data={"schema": "{not valid json"},
    )
    assert resp.status_code == 422
    assert "schema" in resp.json()["detail"]


def test_route_maps_unrecoverable_response_to_400(client, sample_pdf):
    from backend.services.extract_response import ExtractResponseError

    async def bad_response(messages):
        raise ExtractResponseError("could not recover JSON")

    with patch(
        "backend.main.parser_service.parse", new=AsyncMock(side_effect=bad_response)
    ):
        resp = client.post(
            "/extract",
            files={"file": ("invoice.pdf", sample_pdf, "application/pdf")},
            data={"schema": json.dumps(SCHEMA)},
        )
    assert resp.status_code == 400
    assert "recover" in resp.json()["detail"]


def test_route_maps_transient_failure_to_500(client, sample_pdf):
    async def network_error(messages):
        raise RuntimeError("network hiccup")

    with patch(
        "backend.main.parser_service.parse", new=AsyncMock(side_effect=network_error)
    ):
        resp = client.post(
            "/extract",
            files={"file": ("invoice.pdf", sample_pdf, "application/pdf")},
            data={"schema": json.dumps(SCHEMA)},
        )
    assert resp.status_code == 500
