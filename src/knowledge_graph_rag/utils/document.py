# SPDX-FileCopyrightText: 2026 Haruno YUZUNOKI
# SPDX-License-Identifier: (MIT OR Beerware)

from typing import NamedTuple


class Paragraph(NamedTuple):
    """文書内のパラグラフ情報

    Attributes:
        id (str): パラグラフの識別ID
        text (str): パラグラフの本文
        section_name (str): パラグラフが属するセクション名
        doc_id (str): パラグラフの参照元となる文書ID
    """

    id: str
    text: str
    section_name: str
    doc_id: str


class Title(NamedTuple):
    """文書のタイトル情報

    Attributes:
        id (str): 文書の識別ID。
        test (str): 文書のタイトル
    """

    id: str
    test: str


class Query(NamedTuple):
    """クエリと回答、参照先の文書情報

    Attributes:
        id (str): 質問の識別ID
        query (str): 質問文
        answer (str | None): 質問に対する回答(回答不能の場合はNone)
        doc_ids (list[str]): 質問の参照先となる文書IDのリスト
        dataset (str): 質問が属するデータセットのラベル(train/dev/test)
    """

    id: str
    query: str
    answer: str | None
    doc_ids: list[str]
    dataset: str
