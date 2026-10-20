# SPDX-FileCopyrightText: 2026 Haruno YUZUNOKI
# SPDX-License-Identifier: (MIT OR Beerware)

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

PathStr = Path | str


@dataclass(frozen=True)
class DataFormat:
    """データ形式の基底クラス。"""

    @classmethod
    def from_json(cls, filename: PathStr) -> Self:
        """JSONファイルからデータを取得する関数

        Args:
            filename: 入力JSONファイルのパス

        Returns:
            Self: JSONデータから取得したデータ
        """
        with open(filename, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(cls, filename: PathStr | None = None) -> str:
        """JSONフォーマットでデータを出力する関数

        Args:
            filename: 出力JSONファイルのパス(Noneの場合、ファイル出力しない)

        Returns:
            str: JSONデータのJSONフォーマット
        """
        json_format = json.dumps(asdict(cls), ensure_ascii=False)
        # save JSON file.
        if filename is not None:
            with open(filename, mode="w", encoding="utf-8") as f:
                f.write(json_format)
        return json_format


@dataclass(frozen=True)
class Paragraph(DataFormat):
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


@dataclass(frozen=True)
class Title(DataFormat):
    """文書のタイトル情報

    Attributes:
        id (str): 文書の識別ID。
        text (str): 文書のタイトル
    """

    id: str
    text: str


@dataclass(frozen=True)
class Query(DataFormat):
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
