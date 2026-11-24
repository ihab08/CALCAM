import argparse
from pathlib import Path

import numpy as np

from src.cka import linear_cka


# ============================================================
# Utilities
# ============================================================

def load_npy(path):
    return np.load(path, allow_pickle=False)


def clean_and_standardize(X, eps=1e-6):
    X = np.asarray(X, dtype=np.float64)

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)

    return (X - mean) / (std + eps)


def mean_pool_if_needed(X):
    """
    Mean-pool temporal embeddings if shape is (N, T, D).
    """
    X = np.asarray(X)

    if X.ndim == 3:
        return np.nanmean(X, axis=1)

    if X.ndim == 2:
        return X

    raise ValueError(
        f"Expected a 2D or 3D representation, got {X.shape}"
    )


def compute_matrix(representations):
    """
    Compute pairwise Linear CKA.

    representations:
        dict {modality_name: numpy array}
    """

    names = list(representations.keys())

    n = len(names)

    matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):

            score = linear_cka(
                representations[names[i]],
                representations[names[j]]
            )

            matrix[i, j] = score
            matrix[j, i] = score

    return names, matrix


def selected_pair(names, matrix):

    upper = np.triu(matrix, k=1)

    i, j = np.unravel_index(
        np.argmax(upper),
        upper.shape
    )

    return (
        names[i],
        names[j],
        matrix[i, j]
    )


# ============================================================
# C-MAPSS
# ============================================================

def load_cmapss(data_dir):

    data_dir = Path(data_dir)

    modalities = {}

    for name, prefix in [
        ("M1", "Z1"),
        ("M2", "Z2"),
        ("M3", "Z3"),
    ]:

        train = load_npy(
            data_dir / f"{prefix}_tr.npy"
        )

        val = load_npy(
            data_dir / f"{prefix}_va.npy"
        )

        X = np.concatenate(
            [train, val],
            axis=0
        )

        modalities[name] = clean_and_standardize(X)

    return modalities


# ============================================================
# FRACTURE
# ============================================================

def load_fracture(data_dir):

    data_dir = Path(data_dir)

    modalities = {}

    configuration = {
        "Image": "Zimg",
        "Text": "Ztxt",
        "Tabular": "Ztab",
    }

    for name, prefix in configuration.items():

        train = load_npy(
            data_dir / f"{prefix}_train.npy"
        )

        val = load_npy(
            data_dir / f"{prefix}_val.npy"
        )

        test = load_npy(
            data_dir / f"{prefix}_test.npy"
        )

        X = np.concatenate(
            [train, val, test],
            axis=0
        )

        modalities[name] = clean_and_standardize(X)

    return modalities


# ============================================================
# CMU-MOSEI
# ============================================================

def load_mosei(data_dir):

    data_dir = Path(data_dir)

    visual = load_npy(
        data_dir / "X_visual_openface.npy"
    )

    audio = load_npy(
        data_dir / "X_audio_covarep.npy"
    )

    text = load_npy(
        data_dir / "X_text_wordvec.npy"
    )

    visual = mean_pool_if_needed(visual)
    audio = mean_pool_if_needed(audio)
    text = mean_pool_if_needed(text)

    return {
        "Visual": clean_and_standardize(visual),
        "Audio": clean_and_standardize(audio),
        "Text": clean_and_standardize(text),
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Pairwise Linear CKA analysis used by CALCAM."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "cmapss",
            "fracture",
            "cmu-mosei"
        ]
    )

    parser.add_argument(
        "--data-dir",
        required=True
    )

    args = parser.parse_args()


    if args.dataset == "cmapss":

        representations = load_cmapss(
            args.data_dir
        )

    elif args.dataset == "fracture":

        representations = load_fracture(
            args.data_dir
        )

    else:

        representations = load_mosei(
            args.data_dir
        )


    names, matrix = compute_matrix(
        representations
    )

    print("\nLinear CKA matrix\n")

    print(
        np.round(
            matrix,
            6
        )
    )


    a, b, score = selected_pair(
        names,
        matrix
    )


    print("\nSelected CALCAM pair")

    print(
        f"{a} -- {b} "
        f"(CKA = {score:.6f})"
    )


if __name__ == "__main__":
    main()
