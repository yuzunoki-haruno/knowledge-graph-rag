import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from knowledge_graph_rag.utils import LocalLLM, load_embedder, load_reranker

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

LLM_DIR = Path(os.environ.get("LLM_DIRECTORY"))
EMBEDDER_DIR = Path(os.environ.get("EMBEDDER_DIRECTORY"))
RERANKER_DIR = Path(os.environ.get("RERANKER_DIRECTORY"))

LLM_PATH_LIST = [p for p in LLM_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]
EMBEDDER_PATH_LIST = [
    p for p in EMBEDDER_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")
]
RERANKER_PATH_LIST = [
    p for p in RERANKER_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")
]


@pytest.mark.parametrize("model_path", LLM_PATH_LIST)
def test_local_llm(model_path: str) -> None:
    llm = LocalLLM(model_path)

    answer = llm.create("Is water wet? Answer yes or no.")
    assert answer.strip().lower() == "yes"

    answer = llm.create("Is water dry? Answer yes or no.")
    assert answer.strip().lower() == "no"


@pytest.mark.parametrize("model_path", EMBEDDER_PATH_LIST)
def test_qwen_embedder(model_path: str, device: str) -> None:
    model = load_embedder(model_path, device=device)
    texts = ["Is water wet? Answer yes or no."]
    embeddings = model.encode(texts, normalize_embeddings=True)
    assert embeddings.shape == (1, 1024)


@pytest.mark.parametrize("model_path", RERANKER_PATH_LIST)
def test_qwen_reranker(model_path: str, device: str) -> None:
    model = load_reranker(model_path, device=device)

    query = "What is the capital of China?"
    documents = [
        "Gravity is a force that attracts two bodies towards each other.",
        "The capital of China is Beijing.",
    ]
    rankings = model.rank(query, documents)
    assert rankings[0]["corpus_id"] == 1 and rankings[1]["corpus_id"] == 0
