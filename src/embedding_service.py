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

LOCAL_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "multilingual-e5-small"
PROJECT_CACHE_DIR = PROJECT_ROOT / "data" / "model_cache_e5_v2"

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "all_chunks.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_PATH = OUTPUT_DIR / "all_embeddings.npz"
INFO_PATH = OUTPUT_DIR / "embedding_info.json"


def average_pool(last_hidden_state, attention_mask):
    """Average real token vectors and ignore padding."""

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    summed_embeddings = torch.sum(last_hidden_state * mask, dim=1)
    token_counts = torch.clamp(mask.sum(dim=1), min=1e-9)

    return summed_embeddings / token_counts


def load_chunks():
    """Load all generated chunks."""

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Chunks file was not found:\n"
            f"{CHUNKS_PATH}\n"
            "Run src/chunker.py first."
        )

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        chunks = data
    elif isinstance(data, dict):
        chunks = data.get("chunks", [])
    else:
        raise ValueError("all_chunks.json must contain a list of chunks.")

    if not chunks:
        raise ValueError("No chunks were found.")

    return chunks


def test_model_candidate(tokenizer, model, device):
    """Ensure a model produces different vectors for different questions."""

    probe_texts = [
        "query: What AP does DETR achieve on COCO?",
        "query: What is the capital city of Japan?",
    ]

    inputs = tokenizer(
        probe_texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    first_ids = inputs["input_ids"][0].cpu().numpy()
    second_ids = inputs["input_ids"][1].cpu().numpy()

    if np.array_equal(first_ids, second_ids):
        raise RuntimeError(
            "Tokenizer produced identical token IDs for different questions."
        )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = average_pool(outputs.last_hidden_state, inputs["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.cpu().numpy().astype(np.float32)

    first_embedding = embeddings[0]
    second_embedding = embeddings[1]

    cosine_similarity = float(np.dot(first_embedding, second_embedding))
    maximum_difference = float(np.max(np.abs(first_embedding - second_embedding)))

    if np.allclose(first_embedding, second_embedding, atol=1e-7):
        raise RuntimeError(
            "Model produced identical embeddings for different questions."
        )

    if cosine_similarity >= 0.9999:
        raise RuntimeError(
            "Model produced almost identical embeddings.\n"
            f"Cosine similarity: {cosine_similarity:.8f}"
        )

    return cosine_similarity, maximum_difference


class EmbeddingService:
    """Generate normalized multilingual E5 embeddings."""

    def __init__(self):
        self.device = "cpu"

        result = self._load_working_model()
        (
            self.tokenizer,
            self.model,
            self.model_source,
            probe_similarity,
            probe_difference,
        ) = result

        self.model.to(self.device)
        self.model.eval()

        hidden_size = int(self.model.config.hidden_size)

        if hidden_size != EMBEDDING_DIMENSION:
            raise ValueError(
                "Unexpected embedding dimension.\n"
                f"Expected: {EMBEDDING_DIMENSION}\n"
                f"Actual: {hidden_size}"
            )

        print(f"Device: {self.device}")
        print(f"Selected model source: {self.model_source}")
        print("Startup embedding test passed.")
        print(f"Probe cosine similarity: {probe_similarity:.8f}")
        print(f"Probe maximum difference: {probe_difference:.8f}")

    def _load_tokenizer(self, model_location, cache_dir=None):
        """Try fast tokenizer, then slow tokenizer."""

        errors = []

        for use_fast in [True, False]:
            try:
                tokenizer_kwargs = {
                    "local_files_only": True,
                    "use_fast": use_fast,
                }

                if cache_dir is not None:
                    tokenizer_kwargs["cache_dir"] = str(cache_dir)

                tokenizer = AutoTokenizer.from_pretrained(
                    model_location,
                    **tokenizer_kwargs,
                )

                return tokenizer

            except Exception as error:
                errors.append(f"use_fast={use_fast}: {error}")

        raise RuntimeError("Tokenizer could not be loaded.\n" + "\n".join(errors))

    def _load_working_model(self):
        """Try all available model locations."""

        candidates = []

        if LOCAL_MODEL_PATH.exists():
            candidates.append(
                {
                    "name": "local model folder",
                    "location": str(LOCAL_MODEL_PATH),
                    "cache_dir": None,
                }
            )

        if PROJECT_CACHE_DIR.exists():
            candidates.append(
                {
                    "name": "project Hugging Face cache",
                    "location": MODEL_NAME,
                    "cache_dir": PROJECT_CACHE_DIR,
                }
            )

        candidates.append(
            {
                "name": "default Hugging Face cache",
                "location": MODEL_NAME,
                "cache_dir": None,
            }
        )

        errors = []

        for candidate in candidates:
            candidate_name = candidate["name"]

            print()
            print(f"Trying model source: {candidate_name}")

            try:
                tokenizer = self._load_tokenizer(
                    model_location=candidate["location"],
                    cache_dir=candidate["cache_dir"],
                )

                model_kwargs = {"local_files_only": True}

                if candidate["cache_dir"] is not None:
                    model_kwargs["cache_dir"] = str(candidate["cache_dir"])

                model = AutoModel.from_pretrained(
                    candidate["location"],
                    **model_kwargs,
                )

                probe_similarity, probe_difference = test_model_candidate(
                    tokenizer=tokenizer,
                    model=model,
                    device=self.device,
                )

                return (
                    tokenizer,
                    model,
                    candidate_name,
                    probe_similarity,
                    probe_difference,
                )

            except Exception as error:
                errors.append(f"{candidate_name}:\n{error}")
                print(f"  Failed: {error}")

        error_text = "\n\n".join(errors)

        raise RuntimeError(
            "No valid multilingual-e5-small model could be loaded.\n\n"
            "The model must exist in one of these places:\n"
            f"1. {LOCAL_MODEL_PATH}\n"
            f"2. {PROJECT_CACHE_DIR}\n"
            "3. The default Hugging Face cache\n\n"
            f"Details:\n{error_text}"
        )

    def check_lengths(self, texts, text_ids=None):
        """Check input lengths before embedding."""

        longest = 0
        oversized = []

        for index, text in enumerate(texts):
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )

            token_count = len(token_ids)
            longest = max(longest, token_count)

            if token_count > MAX_LENGTH:
                item_id = text_ids[index] if text_ids is not None else str(index)
                oversized.append((item_id, token_count))

        print(f"Maximum input length: {longest}/{MAX_LENGTH} tokens")

        if oversized:
            for item_id, token_count in oversized[:20]:
                print(f"  {item_id}: {token_count} tokens")

            raise ValueError(
                f"{len(oversized)} inputs exceed the {MAX_LENGTH}-token limit."
            )

    def _encode_batch(self, texts):
        """Encode one batch."""

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        sequence_length = int(inputs["input_ids"].shape[1])

        if sequence_length > MAX_LENGTH:
            raise ValueError(
                f"Batch length is {sequence_length}, but the limit is {MAX_LENGTH}."
            )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = average_pool(outputs.last_hidden_state, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)

    def _encode(self, texts, text_ids=None):
        """Encode texts in batches."""

        if not isinstance(texts, list) or not texts:
            raise ValueError("texts must be a non-empty list.")

        self.check_lengths(texts, text_ids)

        all_embeddings = []

        for start in range(0, len(texts), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(texts))
            batch = texts[start:end]

            batch_embeddings = self._encode_batch(batch)
            all_embeddings.append(batch_embeddings)

            print(f"Embedded: {end}/{len(texts)}")

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(f"Wrong embedding shape: {embeddings.shape}")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain NaN or Inf.")

        norms = np.linalg.norm(embeddings, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("Embeddings are not normalized.")

        return embeddings

    def encode_documents(self, texts, chunk_ids=None):
        """Generate E5 passage embeddings."""

        passages = [f"passage: {text.strip()}" for text in texts]
        return self._encode(passages, chunk_ids)

    def encode_query(self, question):
        """Generate one E5 query embedding."""

        if not isinstance(question, str):
            raise ValueError("Question must be a string.")

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        return self._encode([f"query: {question}"])[0]


def build_embeddings():
    """Generate and save embeddings for all chunks."""

    chunks = load_chunks()

    chunk_ids = []
    embedding_texts = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        embedding_text = chunk.get("embedding_text")

        if not chunk_id:
            raise ValueError("A chunk has no chunk_id.")

        if not isinstance(embedding_text, str) or not embedding_text.strip():
            raise ValueError(f"Chunk {chunk_id} has no embedding_text.")

        chunk_ids.append(str(chunk_id))
        embedding_texts.append(embedding_text.strip())

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs were found.")

    print(f"Number of chunks: {len(chunks)}")

    service = EmbeddingService()
    embeddings = service.encode_documents(embedding_texts, chunk_ids)

    expected_shape = (len(chunk_ids), EMBEDDING_DIMENSION)

    if embeddings.shape != expected_shape:
        raise ValueError(
            "Wrong embedding shape.\n"
            f"Expected: {expected_shape}\n"
            f"Actual: {embeddings.shape}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embeddings,
        chunk_ids=np.array(chunk_ids, dtype=str),
    )

    norms = np.linalg.norm(embeddings, axis=1)

    information = {
        "model_name": MODEL_NAME,
        "model_source": service.model_source,
        "embedding_count": len(chunk_ids),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "normalized": True,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "minimum_vector_norm": float(norms.min()),
        "maximum_vector_norm": float(norms.max()),
    }

    with INFO_PATH.open("w", encoding="utf-8") as file:
        json.dump(information, file, ensure_ascii=False, indent=2)

    print()
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Minimum vector norm: {norms.min():.6f}")
    print(f"Maximum vector norm: {norms.max():.6f}")
    print(f"Saved embeddings: {EMBEDDINGS_PATH}")
    print(f"Saved information: {INFO_PATH}")
    print("Embedding generation completed successfully.")


if __name__ == "__main__":
    build_embeddings()
