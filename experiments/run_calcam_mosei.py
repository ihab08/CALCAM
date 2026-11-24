"""
Train and evaluate CALCAM on CMU-MOSEI.

Expected local files
--------------------
X_visual_openface.npy
X_audio_covarep.npy
X_text_wordvec.npy
sentiment.npy

The sentiment labels are converted into binary labels:
    0 = negative sentiment
    1 = non-negative sentiment

Linear CKA selects Visual--Text as the most aligned modality pair.
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.calcam import CALCAM


SEEDS = [0, 1, 2]
SPLIT_SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Loading and preprocessing
# ============================================================

def load_array(directory, filename):
    return np.load(
        Path(directory) / filename,
        allow_pickle=False,
    )


def mean_pool_if_needed(array):
    """
    Convert temporal representations (N, T, D) to (N, D).

    Representations already stored as (N, D) are unchanged.
    """
    array = np.asarray(array)

    if array.ndim == 3:
        return np.nanmean(
            array,
            axis=1,
        )

    if array.ndim == 2:
        return array

    raise ValueError(
        "Expected a 2D or 3D representation, "
        f"but received shape {array.shape}."
    )


def clean_and_standardize(array, eps=1e-6):
    """
    Replace non-finite values and standardize each feature.
    """
    array = np.asarray(
        array,
        dtype=np.float64,
    )

    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mean = array.mean(
        axis=0,
        keepdims=True,
    )

    standard_deviation = array.std(
        axis=0,
        keepdims=True,
    )

    standardized = (
        array - mean
    ) / (
        standard_deviation + eps
    )

    return standardized.astype(
        np.float32
    )


def load_mosei(data_dir):
    visual = load_array(
        data_dir,
        "X_visual_openface.npy",
    )

    audio = load_array(
        data_dir,
        "X_audio_covarep.npy",
    )

    text = load_array(
        data_dir,
        "X_text_wordvec.npy",
    )

    sentiment = load_array(
        data_dir,
        "sentiment.npy",
    ).astype(np.float32)

    visual = mean_pool_if_needed(
        visual
    )

    audio = mean_pool_if_needed(
        audio
    )

    text = mean_pool_if_needed(
        text
    )

    if not (
        len(visual)
        == len(audio)
        == len(text)
        == len(sentiment)
    ):
        raise ValueError(
            "Visual, audio, text, and labels must contain "
            "the same number of aligned utterances."
        )

    visual = clean_and_standardize(
        visual
    )

    audio = clean_and_standardize(
        audio
    )

    text = clean_and_standardize(
        text
    )

    # Binary sentiment:
    # negative = 0
    # non-negative = 1
    labels = (
        sentiment >= 0.0
    ).astype(np.float32)

    return (
        visual,
        audio,
        text,
        labels,
    )


# ============================================================
# Fixed train / validation / test split
# ============================================================

def split_data(
    visual,
    audio,
    text,
    labels,
):
    indices = np.arange(
        len(labels)
    )

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=SPLIT_SEED,
        stratify=labels,
    )

    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=0.20,
        random_state=SPLIT_SEED,
        stratify=labels[train_indices],
    )

    def take(array, indices_to_keep):
        return array[indices_to_keep]

    train = (
        take(visual, train_indices),
        take(audio, train_indices),
        take(text, train_indices),
        take(labels, train_indices),
    )

    validation = (
        take(visual, val_indices),
        take(audio, val_indices),
        take(text, val_indices),
        take(labels, val_indices),
    )

    test = (
        take(visual, test_indices),
        take(audio, test_indices),
        take(text, test_indices),
        take(labels, test_indices),
    )

    return train, validation, test


# ============================================================
# DataLoader
# ============================================================

def create_loader(
    visual,
    audio,
    text,
    labels,
    batch_size,
    shuffle,
):
    dataset = TensorDataset(
        torch.from_numpy(visual),
        torch.from_numpy(audio),
        torch.from_numpy(text),
        torch.from_numpy(labels),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    targets = []
    probabilities = []

    for visual, audio, text, labels in loader:
        visual = visual.to(device)
        audio = audio.to(device)
        text = text.to(device)

        logits = model(
            [
                visual,
                audio,
                text,
            ]
        )

        probability = torch.sigmoid(
            logits
        )

        targets.append(
            labels.numpy()
        )

        probabilities.append(
            probability.cpu().numpy()
        )

    y_true = np.concatenate(
        targets
    ).astype(int)

    y_probability = np.concatenate(
        probabilities
    )

    y_prediction = (
        y_probability >= 0.5
    ).astype(int)

    probability_matrix = np.column_stack(
        [
            1.0 - y_probability,
            y_probability,
        ]
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            y_prediction,
        ),

        "micro_f1": f1_score(
            y_true,
            y_prediction,
            average="micro",
            zero_division=0,
        ),

        "macro_f1": f1_score(
            y_true,
            y_prediction,
            average="macro",
            zero_division=0,
        ),

        "log_loss": log_loss(
            y_true,
            probability_matrix,
            labels=[0, 1],
        ),
    }


# ============================================================
# Training
# ============================================================

def train_one_seed(
    seed,
    train_loader,
    validation_loader,
    test_loader,
    input_dims,
    positive_weight,
    device,
):
    set_seed(seed)

    # Modality ordering:
    #
    # 0 = Visual
    # 1 = Audio
    # 2 = Text
    #
    # Linear CKA selects Visual--Text.
    # Audio is processed as the remaining modality.

    model = CALCAM(
        input_dims=input_dims,
        selected_pair=(0, 2),
        d_model=256,
        nhead=8,
        num_shared_layers=1,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=1,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    maximum_epochs = 25
    patience = 6

    best_macro_f1 = -1.0
    best_state = None
    wait = 0

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):
        model.train()

        for visual, audio, text, labels in train_loader:
            visual = visual.to(device)
            audio = audio.to(device)
            text = text.to(device)

            labels = (
                labels
                .to(device)
                .float()
                .view(-1)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                [
                    visual,
                    audio,
                    text,
                ]
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
        )

        print(
            f"[seed={seed}] "
            f"epoch={epoch:02d} | "
            f"val acc={validation_metrics['accuracy']:.3f} | "
            f"val macro-F1={validation_metrics['macro_f1']:.3f}"
        )

        current_score = validation_metrics[
            "macro_f1"
        ]

        if current_score > best_macro_f1 + 1e-6:
            best_macro_f1 = current_score

            best_state = copy.deepcopy(
                model.state_dict()
            )

            wait = 0

        else:
            wait += 1

            if wait >= patience:
                print(
                    f"[seed={seed}] early stopping"
                )
                break

    if best_state is None:
        raise RuntimeError(
            "No valid checkpoint was produced."
        )

    model.load_state_dict(
        best_state
    )

    return evaluate(
        model,
        test_loader,
        device,
    )


# ============================================================
# Summary
# ============================================================

def summarize(results):
    metrics = [
        "accuracy",
        "micro_f1",
        "macro_f1",
        "log_loss",
    ]

    print(
        "\n================================"
    )

    print(
        "CALCAM - CMU-MOSEI"
    )

    print(
        "TEST RESULTS: mean ± std"
    )

    print(
        "================================"
    )

    for metric in metrics:
        values = np.array(
            [
                result[metric]
                for result in results
            ],
            dtype=float,
        )

        print(
            f"{metric:>10}: "
            f"{values.mean():.3f} "
            f"± {values.std(ddof=0):.3f}"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        required=True,
        help=(
            "Directory containing the CMU-MOSEI "
            "utterance-level representations."
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    visual, audio, text, labels = load_mosei(
        args.data_dir
    )

    train, validation, test = split_data(
        visual,
        audio,
        text,
        labels,
    )

    (
        visual_train,
        audio_train,
        text_train,
        labels_train,
    ) = train

    (
        visual_validation,
        audio_validation,
        text_validation,
        labels_validation,
    ) = validation

    (
        visual_test,
        audio_test,
        text_test,
        labels_test,
    ) = test

    print(
        "Train:",
        visual_train.shape,
        audio_train.shape,
        text_train.shape,
        labels_train.shape,
    )

    print(
        "Validation:",
        visual_validation.shape,
        audio_validation.shape,
        text_validation.shape,
        labels_validation.shape,
    )

    print(
        "Test:",
        visual_test.shape,
        audio_test.shape,
        text_test.shape,
        labels_test.shape,
    )

    positives = float(
        labels_train.sum()
    )

    negatives = float(
        len(labels_train)
        - positives
    )

    positive_weight = torch.tensor(
        [
            negatives
            / (
                positives
                + 1e-8
            )
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        "Positive weight:",
        float(
            positive_weight.item()
        )
    )

    train_loader = create_loader(
        visual_train,
        audio_train,
        text_train,
        labels_train,
        batch_size=128,
        shuffle=True,
    )

    validation_loader = create_loader(
        visual_validation,
        audio_validation,
        text_validation,
        labels_validation,
        batch_size=256,
        shuffle=False,
    )

    test_loader = create_loader(
        visual_test,
        audio_test,
        text_test,
        labels_test,
        batch_size=256,
        shuffle=False,
    )

    input_dims = [
        visual.shape[1],
        audio.shape[1],
        text.shape[1],
    ]

    print(
        "Input dimensions:",
        input_dims
    )

    results = []

    for seed in SEEDS:
        metrics = train_one_seed(
            seed=seed,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            input_dims=input_dims,
            positive_weight=positive_weight,
            device=device,
        )

        results.append(
            metrics
        )

    summarize(
        results
    )


if __name__ == "__main__":
    main()
