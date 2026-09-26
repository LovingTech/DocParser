import os
from pathlib import Path
from functools import lru_cache

_FILE_DIRECTORY = Path(__file__).resolve().parent


@lru_cache
def get_prompt(name: str) -> str:
    path = f"{_FILE_DIRECTORY}/{name}.md"
    if not os.path.exists(path):
        raise ValueError(f"prompt {name} doesn't exist at {path}")

    with open(path, "r") as file:
        return file.read()
