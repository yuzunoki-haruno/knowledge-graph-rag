from typing import NamedTuple


class Paragraph(NamedTuple):
    id: str
    text: str
    section_name: str
    doc_id: str


class Title(NamedTuple):
    id: str
    test: str


class Query(NamedTuple):
    id: str
    query: str
    answer: str | None
    doc_ids: list[str]
    dataset: str
