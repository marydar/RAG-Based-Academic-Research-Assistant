import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSION = 384
MAX_LENGTH = 512
BATCH_SIZE = 4

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "all_chunks.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_PATH = OUTPUT_DIR / "all_embeddings.npz"
INFO_PATH = OUTPUT_DIR / "embedding_info.json"


def average_pool(last_hidden_state, attention_mask):
    """از بردار توکن‌های واقعی، میانگین می‌گیرد."""

    mask = attention_mask.unsqueeze(-1).bool()

    last_hidden_state = last_hidden_state.masked_fill(
        ~mask,
        0.0,
    )

    return (
        last_hidden_state.sum(dim=1)
        / attention_mask.sum(dim=1).unsqueeze(-1)
    )


def load_chunks():
    """Chunkها را از فایل JSON می‌خواند."""

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        chunks = data.get("chunks", [])
    else:
        chunks = data

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("No chunks were found.")

    return chunks


class EmbeddingService:
    """Embedding اسناد و سؤال را با مدل E5 تولید می‌کند."""

    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("Loading tokenizer from local cache...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        print("Loading model from local cache...")

        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"Device: {self.device}")

    def _encode(self, texts):
        """یک لیست از متن‌ها را به Embedding تبدیل می‌کند."""

        all_embeddings = []

        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]

            inputs = self.tokenizer(
                batch,
                max_length=MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = average_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            )

            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1,
            )

            all_embeddings.append(
                embeddings.cpu().numpy()
            )

            completed = min(
                start + BATCH_SIZE,
                len(texts),
            )

            print(
                f"Embedded: {completed}/{len(texts)}"
            )

        return np.vstack(all_embeddings).astype(
            np.float32
        )

    def encode_documents(self, texts):
        """برای Chunkهای مقالات Embedding تولید می‌کند."""

        if not texts:
            raise ValueError(
                "Document texts cannot be empty."
            )

        passages = [
            f"passage: {text}"
            for text in texts
        ]

        return self._encode(passages)

    def encode_query(self, question):
        """برای سؤال کاربر Embedding تولید می‌کند."""

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        queries = [
            f"query: {question.strip()}"
        ]

        return self._encode(queries)[0]


def build_embeddings():
    """Embedding تمام Chunkها را تولید و ذخیره می‌کند."""

    chunks = load_chunks()

    chunk_ids = []
    texts = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("embedding_text")

        if not chunk_id:
            raise ValueError(
                "A chunk has no chunk_id."
            )

        if not text or not text.strip():
            raise ValueError(
                f"Chunk {chunk_id} has no embedding_text."
            )

        chunk_ids.append(str(chunk_id))
        texts.append(text.strip())

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Duplicate chunk IDs were found."
        )

    print(f"Number of chunks: {len(chunks)}")

    service = EmbeddingService()
    embeddings = service.encode_documents(texts)

    expected_shape = (
        len(chunks),
        EMBEDDING_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Wrong shape: {embeddings.shape}. "
            f"Expected: {expected_shape}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embeddings,
        chunk_ids=np.array(chunk_ids),
    )

    info = {
        "model_name": MODEL_NAME,
        "embedding_count": len(chunk_ids),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "normalized": True,
        "max_length": MAX_LENGTH,
    }

    with open(INFO_PATH, "w", encoding="utf-8") as file:
        json.dump(
            info,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved embeddings: {EMBEDDINGS_PATH}")
    print(f"Saved information: {INFO_PATH}")


if __name__ == "__main__":
    build_embeddings()