"""Request and response models."""

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class FieldType(str, Enum):
    """Data types accepted in the extraction schema."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"


class FieldSchema(BaseModel):
    """A single element to extract from the document.

    - name: the key used in the output JSON.
    - type: the data type of the value.
    - description: what the element refers to, used to guide the model.
    """

    name: str
    type: FieldType = FieldType.STRING
    description: str = ""


class Message(BaseModel):
    role: str
    content: str | Any


class Schema(BaseModel):
    """Named group of fields. The schema name is echoed back in the response."""

    name: str = ""
    fields: List[FieldSchema] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    request_id: str
    status: str = "success"
    schema_name: str = ""
    data: Dict[str, Any]
