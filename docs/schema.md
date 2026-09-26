# Schema reference

A **schema** tells the parser *what* to extract from a document and *how* to
structure the result. You send it to `/extract` or `/extract/json` and the VLM
returns an object whose keys match your fields.

```json
{
  "name": "invoice",
  "fields": [
    { "name": "invoice_number", "type": "string", "description": "Invoice number" },
    { "name": "total", "type": "number", "description": "Grand total due" }
  ]
}
```

## The schema object

| Field    | Type        | Required | Description                                            |
|----------|-------------|----------|--------------------------------------------------------|
| `name`   | `string`    | No       | Optional label for the schema, echoed back in responses. |
| `fields` | `Field[]`   | No       | The fields to extract, in the order you want them returned. |

## A field

| Field         | Type           | Required | Description                                             |
|---------------|----------------|----------|---------------------------------------------------------|
| `name`        | `string`       | Yes      | Key used in the output JSON.                            |
| `type`        | `FieldType`    | No       | Data type (see table below). Default `string`.          |
| `description` | `string`       | No       | Free-text guide for the model. Use it to pin down formats (e.g. `YYYY-MM-DD`). |
| `items`       | `Field[]`      | No       | Nested fields describing each entry of an `array` field. Empty for scalars. |

### Field types

| Type        | JSON value                                   | Notes |
|-------------|----------------------------------------------|-------|
| `string`    | `"hello"`                                    | Default. Any text value. |
| `integer`   | `42`                                         | Whole numbers only. |
| `number`    | `12.50`                                      | Decimal/fractional amounts. |
| `boolean`   | `true` / `false`                             | Yes/no flags. |
| `date`      | `"2026-09-26"`                               | Normalize to an ISO date when possible. |
| `object`    | `{ ... }`                                    | A single complex value. Use `items` to name its keys. |
| `array`     | `[ ... ]`                                    | A list. Use `items` to describe each entry (see below). |
| `any`       | anything                                     | Don't coerce; return exactly what's found. |

## Repeated data with `array` + `items`

Documents are full of repeated structures — line items, table rows, contact
entries. For these, use `type: "array"` with an `items` list. Each item is a
`Field`, and the model returns a JSON array whose elements are objects carrying
exactly those item fields.

```json
{
  "name": "invoice",
  "fields": [
    { "name": "invoice_number", "type": "string", "description": "Invoice number" },
    {
      "name": "line_items",
      "type": "array",
      "description": "Each row of the items table",
      "items": [
        { "name": "item",   "type": "string",  "description": "Item description" },
        { "name": "qty",    "type": "integer", "description": "Quantity" },
        { "name": "unit_price", "type": "number", "description": "Price per unit" },
        { "name": "amount", "type": "number",  "description": "Line total" }
      ]
    }
  ]
}
```

Expected output:

```json
{
  "invoice_number": "INV-1001",
  "line_items": [
    { "item": "Consulting", "qty": 10, "unit_price": 150.00, "amount": 1500.00 },
    { "item": "Materials",  "qty": 3,  "unit_price": 80.00,  "amount": 240.00 }
  ]
}
```

Rules the model follows for arrays:

- One object per occurrence — never merge rows into a single object.
- Every object contains exactly the `items` fields, in that order.
- Missing values become `null` rather than being omitted.
- Keep the same shape for every entry.

`object` uses the same `items` mechanism for a single (non-listed) complex
value — e.g. a nested address with `street`, `city`, `postal_code` keys.

## Output shape

For a top-level field `foo`, the response `data` object contains a `foo` key of
the requested type. An omitted field simply doesn't appear. Errors during
parsing are surfaced by the endpoint (`400` on bad JSON, `500` on VLM failure).

## Validating a schema

A schema is just JSON validated against `Schema` / `FieldSchema`
(`backend/models.py`). Common mistakes:

- `array` or `object` type without `items` — the value has no defined shape.
- A duplicate `name` across fields — the output key becomes ambiguous.
- An unknown `type` — rejected as a `422`/validation error.
