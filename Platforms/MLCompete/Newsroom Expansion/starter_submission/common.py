from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np


N_FEATURES = 2048
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _bucket(term: str) -> int:
    digest = hashlib.blake2b(
        term.encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") % N_FEATURES


def vectorize_texts(texts) -> np.ndarray:
    """Transformă textele în reprezentarea feature-hashing folosită de model."""
    X = np.zeros((len(texts), N_FEATURES), dtype=np.float32)

    for i, text in enumerate(texts):
        tokens = TOKEN_RE.findall(str(text).lower())
        features = tokens + [
            tokens[j] + "__" + tokens[j + 1]
            for j in range(len(tokens) - 1)
        ]

        for term in features:
            X[i, _bucket(term)] += 1.0

    X = np.log1p(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X /= np.maximum(norms, 1e-8)
    return X


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def predict_labels(model, X: np.ndarray) -> np.ndarray:
    scores = X @ model["W"] + model["b"]
    return softmax(scores).argmax(axis=1)


def load_model(model_dir: str | Path):
    """Încarcă numai parametrii necesari pentru predicție."""
    model_dir = Path(model_dir)

    with (model_dir / "state.pkl").open("rb") as fin:
        state = pickle.load(fin)

    return {
        "W": np.asarray(state["W"]).copy(),
        "b": np.asarray(state["b"]).copy(),
    }


def save_model(
    output_dir: str | Path,
    model,
    metadata=None,
) -> None:
    """Salvează modelul în formatul utilizat de evaluator."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "state.pkl").open("wb") as fout:
        pickle.dump(
            {
                "W": np.asarray(model["W"]),
                "b": np.asarray(model["b"]),
            },
            fout,
            protocol=4,
        )

    (output_dir / "meta.json").write_text(
        json.dumps(metadata or {}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
