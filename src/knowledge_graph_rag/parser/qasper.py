# SPDX-FileCopyrightText: 2026 Haruno YUZUNOKI
# SPDX-License-Identifier: (MIT OR Beerware)

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from knowledge_graph_rag.utils import Paragraph, Query, Title

PathStr = Path | str

WHITE_SPACE = " "
NEW_LINE = "\n"
ABSTRACT_INDEX = 0
PARAGRAPH_INDEX = ABSTRACT_INDEX + 1


def get_paragraphs(record: dict[str, Any]) -> Generator[Paragraph, None, None]:
    """文書データから各セクションのパラグラフ情報を取得する関数

    Args:
        record (dict[str, Any]): 文書データ (文書 ID と本文のセクション情報を含む)

    Yields:
        Generator[Paragraph, None, None]: 文書データの全パラグラフ情報
    """
    doc_id = record["doc_id"]
    for i, text in enumerate(record["full_text"]):
        index = PARAGRAPH_INDEX + i
        paragraph = WHITE_SPACE.join(text["paragraphs"])
        yield Paragraph(f"{doc_id}.{index:04}", paragraph, text["section_name"], doc_id)


def get_abstract(record: dict[str, Any]) -> Paragraph:
    """文書データからAbstractをパラグラフ情報として取得する関数

    Args:
        record (dict[str, Any]): 文書データ (文書 ID と本文のセクション情報を含む)

    Returns:
        Paragraph: Abstractのパラグラフ情報
    """
    doc_id = record["doc_id"]
    return Paragraph(f"{doc_id}.{ABSTRACT_INDEX:04}", record["abstract"], "abstract", doc_id)


def get_answer(record: dict[str, Any]) -> str | None:
    """回答データから回答テキストを取得する関数

    Args:
        record (dict[str, Any]): 回答データ(回答可否や回答内容を含む)

    Returns:
        str | None: 回答テキスト(Yes/No、抽出型回答、自由記述回答)。回答不能の場合は None。
    """
    if record.get("unanswerable", False):
        return None
    if record.get("yes_no") is not None:
        return "Yes" if record["yes_no"] else "No"
    if record.get("extractive_spans"):
        return " ".join(record["extractive_spans"])
    return record.get("free_form_answer")


def get_queries(doc_id: str, label: str, record: dict[str, Any]) -> Generator[Query, None, None]:
    """質問・回答データからクエリ情報を取得する関数

    Args:
        doc_id (str): 質問の参照先となる文書ID。
        label (str): データセットのラベル(train/dev/test)
        record (dict[str, Any]): 質問と回答のデータ

    Yields:
        Generator[Query, None, None]: 質問・回答データの全クエリ情報
    """
    for q_and_a in record["answers"]:
        yield Query(
            record["question_id"],
            record["question"],
            get_answer(q_and_a["answer"]),
            [doc_id],
            label,
        )


def extract_paragraph(record: dict[str, Any], output_dir: Path) -> None:
    """文書データから全パラグラフ情報を抽出し、JSONL形式で保存する関数

    Args:
        record (dict[str, Any]): 文書データ
        output_dir (Path): 出力先ディレクトリ(存在しない場合は作成)
    """
    doc_id = record["doc_id"]
    paragraphs = list(get_paragraphs(record))
    paragraphs.append(get_abstract(record))
    # save file
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"corpus.{doc_id}.jsonl"
    with open(filename, mode="w", encoding="utf-8") as f:
        f.writelines([
            json.dumps(paragraph._asdict(), ensure_ascii=False) + NEW_LINE
            for paragraph in paragraphs
        ])


def extract_title(record: dict[str, Any], output_dir: Path) -> None:
    """文書データからタイトル情報を抽出し、JSONL形式で保存する関数

    Args:
        record (dict[str, Any]): 文書データ
        output_dir (Path): 出力先ディレクトリ(存在しない場合は作成)
    """
    doc_id = record["doc_id"]
    title = Title(doc_id, record["title"])
    # save file
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"title.{doc_id}.jsonl"
    with open(filename, mode="w", encoding="utf-8") as f:
        f.write(json.dumps(title._asdict(), ensure_ascii=False))


def extract_query(record: dict[str, Any], output_dir: Path) -> None:
    """文書データから質問・回答を抽出し、JSONL形式で保存する関数

    Args:
        record (dict[str, Any]): 文書データ
        output_dir (Path): 出力先ディレクトリ(存在しない場合は作成)
    """
    doc_id = record["doc_id"]
    label = record["dataset"]
    queries = []
    for q_and_a in record["qas"]:
        queries.extend(get_queries(doc_id, label, q_and_a))
    # save file
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"question.{doc_id}.jsonl"
    with open(filename, mode="w", encoding="utf-8") as f:
        f.writelines([json.dumps(row._asdict(), ensure_ascii=False) + NEW_LINE for row in queries])


def parse_qasper_dataset(input_root_dir: PathStr, output_root_dir: PathStr) -> None:
    """QASPERデータセットからパラグラフ・タイトル・クエリデータを抽出する関数

    Args:
        input_root_dir (PathStr): 入力JSONファイルを格納したディレクトリのパス
        output_root_dir (PathStr): 抽出したデータの出力先ディレクトリのパス
    """
    if not isinstance(input_root_dir, Path):
        input_root_dir = Path(input_root_dir)
    if not isinstance(output_root_dir, Path):
        output_root_dir = Path(output_root_dir)

    # output directories.
    document_dir = output_root_dir / "documents"
    question_dir = output_root_dir / "questions"
    document_dir.mkdir(parents=True, exist_ok=True)
    question_dir.mkdir(parents=True, exist_ok=True)

    # load raw documents and queries from JSON data file.
    for input_json_path in input_root_dir.glob("*.json"):
        try:
            with open(input_json_path, encoding="utf-8") as json_file:
                raw_dataset = json.load(json_file)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Failed to load JSON: {input_json_path}: {e}")
            continue

        # execute parsing documents and queries.
        for doc_id in tqdm(raw_dataset, desc=f"Preprocessing `{input_json_path.name}`."):
            document: dict = raw_dataset[doc_id]
            document.setdefault("doc_id", doc_id)
            document.setdefault("dataset", input_json_path.stem.split("-")[1])
            extract_title(document, document_dir)
            extract_paragraph(document, document_dir)
            extract_query(document, question_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_root_dir", type=str)
    parser.add_argument("output_root_dir", type=str)
    args = parser.parse_args()

    parse_qasper_dataset(args.input_root_dir, args.output_root_dir)
