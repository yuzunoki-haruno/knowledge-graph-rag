# SPDX-FileCopyrightText: 2026 Haruno YUZUNOKI
# SPDX-License-Identifier: (MIT OR Beerware)
from pathlib import Path
from typing import Any, cast

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

logging.disable_progress_bar()
PathStr = Path | str


class LocalLLM:
    """Hugging FaceのLLMでテキスト生成するクラス

    Attributes:
        llm: LLM。
        tokenizer: LLMに対応するトークナイザー
        max_new_tokens: 生成する最大トークン数
    """

    def __init__(self, model_path: PathStr, max_new_tokens: int = 1024) -> None:
        """モデルとトークナイザーの読み込みと初期化

        Args:
            model_path (PathStr): モデルのファイルパス (Hugging FaceのモデルIDも可)
            max_new_tokens (int, optional): 生成する最大トークン数。デフォルトは1024
        """
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )
        self.max_new_tokens = max_new_tokens

    def create(self, prompt: str) -> str:
        """LLMでプロンプトからテキストを生成

        Args:
            prompt (str): テキスト生成用プロンプト

        Raises:
            RuntimeError: トークナイザーがNoneの場合、デコード結果が文字列でない場合。

        Returns:
            str: 生成テキスト
        """
        if self.tokenizer is None:
            raise RuntimeError("tokenizer is None.")

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to("cuda")
        length = input_ids.shape[-1]

        model = cast(Any, self.llm)
        output = model.generate(input_ids, max_new_tokens=self.max_new_tokens)[0][length:]
        output = self.tokenizer.decode(output, skip_special_tokens=True)
        if not isinstance(output, str):
            raise RuntimeError("decode is failed.")
        return output


def load_embedder(model_path: PathStr, device: str = "cuda") -> SentenceTransformer:
    """埋込モデルを読み込む関数

    Args:
        model_path (PathStr): モデルのファイルパス (Hugging FaceのモデルIDも可)
        device (str, optional): 計算に使用するデバイス(cuda, cpuなど)。 デフォルトは"cuda"

    Returns:
        SentenceTransformer: 埋め込みモデル(SentenceTransformer)
    """
    model_path = str(model_path)
    return SentenceTransformer(model_path, device=device)


def load_reranker(model_path: PathStr, device: str = "cuda") -> CrossEncoder:
    """リランキングモデルを読み込む関数

    Args:
        model_path (PathStr): モデルのファイルパス (Hugging FaceのモデルIDも可)
        device (str, optional): 計算に使用するデバイス(cuda, cpuなど)。 デフォルトは"cuda"

    Returns:
        CrossEncoder: リランキングモデル(CrossEncoder)
    """
    model_path = str(model_path)
    return CrossEncoder(model_path, device=device)
