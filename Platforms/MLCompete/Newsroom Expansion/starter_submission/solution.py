from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from common import save_model, softmax, vectorize_texts


N_CLASSES = 4
NEW_CLASSES = (2, 3)

# hyperparams
EWC_LAMBDA = 1e4
N_ITERS = 600
LR = 0.5


def _load_pretrained_state(pretrained_model_dir: Path) -> dict:
    """Încarcă tot checkpoint-ul Phase 1 (parametri + statistici EWC)."""
    with (pretrained_model_dir / "state.pkl").open("rb") as fin:
        state = pickle.load(fin)

    return {
        "W": np.asarray(state["W"], dtype=np.float64).copy(),
        "b": np.asarray(state["b"], dtype=np.float64).copy(),
        "theta_W": np.asarray(state["theta_W"], dtype=np.float64),
        "theta_b": np.asarray(state["theta_b"], dtype=np.float64),
        "fisher_W": np.asarray(state["fisher_W"], dtype=np.float64),
        "fisher_b": np.asarray(state["fisher_b"], dtype=np.float64),
    }


def _train_ewc(state: dict, X: np.ndarray, y: np.ndarray, seed: int):
    rng = np.random.RandomState(seed)

    W = state["W"].copy()
    b = state["b"].copy()
    theta_W, theta_b = state["theta_W"], state["theta_b"]
    fisher_W, fisher_b = state["fisher_W"], state["fisher_b"]

    W = W + rng.normal(scale=1e-6, size=W.shape)

    n = X.shape[0]
    Y = np.zeros((n, N_CLASSES), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # adam optim
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for t in range(1, N_ITERS + 1):
        logits = X @ W + b
        P = softmax(logits)

        dZ = (P - Y) / n
        gW = X.T @ dZ
        gb = dZ.sum(axis=0)

        gW += EWC_LAMBDA * fisher_W * (W - theta_W)
        gb += EWC_LAMBDA * fisher_b * (b - theta_b)

        # adam step
        mW = beta1 * mW + (1 - beta1) * gW
        vW = beta2 * vW + (1 - beta2) * (gW * gW)
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * (gb * gb)
        mWh = mW / (1 - beta1 ** t); vWh = vW / (1 - beta2 ** t)
        mbh = mb / (1 - beta1 ** t); vbh = vb / (1 - beta2 ** t)
        W -= LR * mWh / (np.sqrt(vWh) + eps)
        b -= LR * mbh / (np.sqrt(vbh) + eps)

    return W.astype(np.float32), b.astype(np.float32)


def adapt(
    phase2_train_csv,
    pretrained_model_dir,
    output_dir,
    seed,
):
    phase2_train_csv = Path(phase2_train_csv)
    pretrained_model_dir = Path(pretrained_model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(phase2_train_csv)
    X = vectorize_texts(train_df["text"].tolist()).astype(np.float64)
    y = train_df["answer"].to_numpy().astype(int)

    state = _load_pretrained_state(pretrained_model_dir)
    W, b = _train_ewc(state, X, y, int(seed))

    save_model(
        output_dir,
        {"W": W, "b": b},
        metadata={
            "format_version": 1,
            "model_type": "hashed-linear-softmax-ewc",
            "n_classes": N_CLASSES,
            "seed": int(seed),
            "ewc_lambda": EWC_LAMBDA,
        },
    )

    manifest = {
        "format_version": 1,
        "model_files": ["state.pkl", "meta.json"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return str(manifest_path)


def _uniform_prior_delta(logits: np.ndarray, iters: int = 200) -> np.ndarray:
    """
    whatever bs a scris claude aicia
    """
    delta = np.zeros(logits.shape[1], dtype=np.float64)
    for _ in range(iters):
        marg = softmax(logits + delta).mean(axis=0)
        delta -= 0.5 * np.log(np.maximum(marg, 1e-12) / (1.0 / logits.shape[1]))
        delta -= delta.mean()
    return delta


def predict(
    test_csv,
    model_dir,
):
    test_csv = Path(test_csv)
    model_dir = Path(model_dir)

    test_df = pd.read_csv(test_csv)

    with (model_dir / "state.pkl").open("rb") as fin:
        model = pickle.load(fin)
    W = np.asarray(model["W"], dtype=np.float64)
    b = np.asarray(model["b"], dtype=np.float64)

    X = vectorize_texts(test_df["text"].tolist()).astype(np.float64)
    logits = X @ W + b
    delta = _uniform_prior_delta(logits)
    predictions = (logits + delta).argmax(axis=1).astype(int)

    return pd.DataFrame(
        {
            "datapointID": test_df["datapointID"].astype(str),
            "subtaskID": test_df["subtaskID"].astype(int),
            "answer": predictions,
        },
        columns=["datapointID", "subtaskID", "answer"],
    )
