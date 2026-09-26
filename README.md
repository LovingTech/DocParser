# Document Parser (VLM)

A small FastAPI service that extracts structured data from PDF documents using an
OpenAI-compatible vision language model (VLM). You send it a PDF and a schema describing
the fields you want; it renders each page to an image and asks the VLM to return those
fields as JSON.

## Architecture

The service layer is split by responsibility so each piece can be reasoned about (and mocked) in isolation:

```text
Document  ──▶  PdfToImageConversionService  ──▶  DocumentParserService  ──▶  LLM Adapter  ──▶  JSON
```

**Domain model**

- `app/services/document.py` — `Document`: the raw PDF bytes plus the `Schema` to extract from.

**Services**

- `PdfToImageConversionService` — renders each PDF page (via PyMuPDF) into a cropped, compressed image.
- `DocumentParserService` — the orchestration / business layer: converts PDFs to images, builds the structured + multi-modal prompt, calls the LLM, and parses the returned JSON back into `data`.
- `LLM` adapter (`app/services/llm.py`) — `OpenAILLM` talks to any OpenAI-compatible VLM (`gpt-4o`, etc.). New backends only need to implement the `chat_completion(messages)` protocol.

**Prompt**

- `app/prompts/parser_system_prompt.md` — the system prompt template that fields are injected into.

## Schema

Each field has:

- `name` — the key used in the output JSON.
- `type` — `string`, `integer`, `number`, `boolean`, `date`, `array`, `object`, or `any`.
- `description` — what the element refers to (guides the model).

Example:

```json
{
  "name": "invoice",
  "fields": [
    { "name": "invoice_number", "type": "string", "description": "Invoice number on the document" },
    { "name": "date", "type": "date", "description": "Invoice issue date" },
    { "name": "total", "type": "number", "description": "Total amount due" },
    { "name": "vendor", "type": "string", "description": "Name of the vendor / supplier" }
  ]
}
```

## Endpoints

- `POST /extract` — multipart/form-data
  - `file`: the PDF binary
  - `schema`: the schema as a JSON string
- `POST /extract/json` — JSON body
  - `file_base64`: base64-encoded PDF
  - `schema`: the schema object
- `GET /health` — health + config info

## Run locally (uv)

```bash
uv sync
uv run uvicorn backend.main:app --reload
```

## Run with Docker

nginx serves the static frontend and proxies the API (`/extract`, `/extract/json`, `/health`) to the FastAPI backend. Start everything with:

```bash
cp .env.example .env   # then set OPENAI_API_KEY
docker compose up --build
```

The frontend is served on the port mapped in `docker-compose.yml` (the `nginx` service `ports` mapping). The backend is not published to the host; reach it only through nginx.

> If port 8000 is already taken on your host (e.g. by a local VLM server), change the nginx `ports` mapping in `docker-compose.yml` (for example to `9000:80`) and use that port.

```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@sample.pdf" \
  -F 'schema={"name":"invoice","fields":[
    {"name":"invoice_number","type":"string","description":"Invoice number"},
    {"name":"total","type":"number","description":"Total amount due"},
    {"name":"date","type":"date","description":"Invoice date"},
    {"name":"vendor","type":"string","description":"Vendor name"}
  ]}'
```

## Configuration (environment variables)

| Variable            | Default                        | Purpose                                             |
|---------------------|--------------------------------|-----------------------------------------------------|
| `OPENAI_API_KEY`    | *(required)*                   | API key for the OpenAI-compatible VLM.              |
| `OPENAI_BASE_URL`   | `https://api.openai.com/v1`    | Override for OpenRouter, vLLM, DeepSeek, etc. In Docker, point this at a host-side VLM with `http://host.docker.internal:8000/v1`. |
| `OPENAI_MODEL`      | `gpt-4o`                       | Vision model (e.g. `gpt-4o`, `gpt-4o-mini`).        |
| `PDF_DPI`           | `200`                          | DPI when rendering each page to an image.           |
| `MAX_PAGES`         | `10`                           | Max pages sent per request.                         |
| `IMAGE_MAX_SIDE`    | `2000`                         | Largest side length of a rendered page image.       |
| `IMAGE_QUALITY`     | `75`                           | JPEG quality for rendered pages (1–100).            |
