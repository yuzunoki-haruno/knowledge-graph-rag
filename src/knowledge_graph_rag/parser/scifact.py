# SPDX-FileCopyrightText: 2026 Haruno YUZUNOKI
# SPDX-License-Identifier: (MIT OR Beerware)

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from knowledge_graph_rag.utils import Paragraph, Query, Title

PathStr = Path | str

NEW_LINE = "\n"
ABSTRACT_INDEX = 0


def get_document(record: dict[str, Any]) -> Paragraph:
    """文書データから各セクションのパラグラフ情報を取得する関数

    Args:
        record (dict[str, Any]): 文書データ (文書 ID と本文のセクション情報を含む)

    Returns:
        Paragraph: 文書データのパラグラフ情報
    """
    doc_id = record["_id"]
    return Paragraph(f"{doc_id}.{ABSTRACT_INDEX:04}", record["text"], "abstract", doc_id)


def load_query_relevance(input_dir: Path) -> pd.DataFrame:
    """クエリと文書の対応関係を取得する関数

    Args:
        input_dir (Path): 入力TSVファイルを格納したディレクトリのパス

    Returns:
        pd.DataFrame: クエリと文書の対応関係
    """
    # train data.
    train_tsv_path = input_dir / "qrels/" / "train.tsv"
    df_train = pd.read_csv(train_tsv_path, sep="\t")
    df_train = pd.concat(
        [df_train, pd.Series("train", index=range(len(df_train)), name="dataset")], axis=1
    )
    # test data.
    test_tsv_path = input_dir / "qrels/" / "test.tsv"
    df_test = pd.read_csv(test_tsv_path, sep="\t")
    df_test = pd.concat(
        [df_test, pd.Series("test", index=range(len(df_test)), name="dataset")], axis=1
    )
    return pd.concat([df_train, df_test], axis=0, ignore_index=True)


def extract_paragraph(record: dict[str, Any], output_dir: Path) -> None:
    """文書データから全パラグラフ情報を抽出し、JSONL形式で保存する関数

    Args:
        record (dict[str, Any]): 文書データ
        output_dir (Path): 出力先ディレクトリ(存在しない場合は作成)
    """
    doc_id = record["_id"]
    records = [get_document(record)]
    # save file
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"corpus.{doc_id}.jsonl"
    with open(filename, mode="w", encoding="utf-8") as f:
        f.writelines([json.dumps(record._asdict()) + NEW_LINE for record in records])


def extract_title(record: dict[str, Any], output_dir: Path) -> None:
    """文書データからタイトル情報を抽出し、JSONL形式で保存する関数

    Args:
        record (dict[str, Any]): 文書データ
        output_dir (Path): 出力先ディレクトリ(存在しない場合は作成)
    """
    doc_id = record["_id"]
    title = Title(doc_id, record["title"])
    # save file
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"title.{doc_id}.jsonl"
    with open(filename, mode="w", encoding="utf-8") as f:
        f.write(json.dumps(title._asdict(), ensure_ascii=False))


def extract_query(record: dict[str, Any], df: pd.DataFrame, output_dir: Path) -> None:
    """文書データから質問・回答を抽出し、JSONL形式で保存する関数

    Args:
        record (dict[str, Any]): 文書データ
        df (pd.DataFrame): クエリと文書の対応関係
        output_dir (Path): 出力先ディレクトリ(存在しない場合は作成)
    """
    question_id = record["_id"]
    df_masked = df[df["query-id"] == int(question_id)]
    doc_ids = df_masked["corpus-id"].to_list()
    dataset = df_masked["dataset"].unique()[0]
    query = Query(question_id, record["text"], None, doc_ids, dataset)
    # save file
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"question.{question_id}.jsonl"
    with open(filename, mode="w", encoding="utf-8") as f:
        f.writelines([json.dumps(query._asdict(), ensure_ascii=False)])


def parse_scifact_dataset(input_root_dir: PathStr, output_root_dir: PathStr) -> None:
    """SciFactデータセットからパラグラフ・タイトル・クエリデータを抽出する関数

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

    # load raw documents from JSON data file.
    input_json_path = input_root_dir / "corpus.jsonl"
    with open(input_json_path, encoding="utf-8") as json_file:
        # get document records.
        lines = list(json_file)
        for line in tqdm(lines, desc=f"Preprocessing `{input_json_path.name}`."):
            try:
                document = json.loads(line)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Failed to load JSON: {input_json_path}: {e}")
                continue

            # execute parsing documents.
            extract_paragraph(document, document_dir)
            extract_title(document, document_dir)

    # load relevance of query to documents.
    df = load_query_relevance(input_root_dir)

    # load raw queries from JSON data file.
    input_json_path = input_root_dir / "queries.jsonl"
    with open(input_json_path, encoding="utf-8") as json_file:
        # get query records.
        lines = list(json_file)
        for line in tqdm(lines, desc=f"Preprocessing `{input_json_path.name}`."):
            try:
                document = json.loads(line)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Failed to load JSON: {input_json_path}: {e}")
                continue

            # execute parsing queries.
            extract_query(document, df, question_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_root_dir", type=str)
    parser.add_argument("output_root_dir", type=str)
    args = parser.parse_args()

    parse_scifact_dataset(args.input_root_dir, args.output_root_dir)
