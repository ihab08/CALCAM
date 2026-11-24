"""
Train and evaluate CALCAM on C-MAPSS.

Expected local files
--------------------
Z1_tr.npy
Z1_va.npy
Z2_tr.npy
Z2_va.npy
Z3_tr.npy
Z3_va.npy
y_tr.npy
y_va.npy

Linear CKA selects M2--M3 as the most aligned modality pair.
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

from torch.utils.data import DataLoader, TensorDataset

from src.calcam import CALCAM


SEEDS = [0, 1, 2]


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
# Loading
# ============================================================

def load_array(directory, filename):
    return np.load(
        Path(directory) / filename,
        allow_pickle=False,
    )


def load_cmapss(data_dir):

    # Train
    z1_train = load_array(
        data_dir,
        "Z1_tr.npy",
    ).astype(np.float32)

    z2_train = load_array(
        data_dir,
        "Z2_tr.npy",
    ).astype(np.float32)

    z3_train = load_array(
        data_dir,
        "Z3_tr.npy",
    ).astype(np.float32)

    y_train = load_array(
        data_dir,
        "y_tr.npy",
    ).astype(np.float32)


    # Validation
    z1_val = load_array(
        data_dir,
        "Z1_va.npy",
    ).astype(np.float32)

    z2_val = load_array(
        data_dir,
        "Z2_va.npy",
    ).astype(np.float32)

    z3_val = load_array(
        data_dir,
        "Z3_va.npy",
    ).astype(np.float32)

    y_val = load_array(
        data_dir,
        "y_va.npy",
    ).astype(np.float32)


    return (
        z1_train,
        z2_train,
        z3_train,
        y_train,

        z1_val,
        z2_val,
        z3_val,
        y_val,
    )


# ============================================================
# DataLoader
# ============================================================

def create_loader(
    z1,
    z2,
    z3,
    labels,
    batch_size,
    shuffle,
):

    dataset = TensorDataset(
        torch.from_numpy(z1),
        torch.from_numpy(z2),
        torch.from_numpy(z3),
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

    for z1, z2, z3, labels in loader:

        z1 = z1.to(device)
        z2 = z2.to(device)
        z3 = z3.to(device)

        logits = model(
            [
                z1,
                z2,
                z3,
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

    y_prob = np.concatenate(
        probabilities
    )

    y_pred = (
        y_prob >= 0.5
    ).astype(int)


    probability_matrix = np.column_stack(
        [
            1.0 - y_prob,
            y_prob,
        ]
    )


    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "micro_f1": f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
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
    val_loader,
    input_dims,
    device,
    positive_weight,
):

    set_seed(seed)

    # --------------------------------------------------------
    # Modality ordering:
    #
    # 0 = M1
    # 1 = M2
    # 2 = M3
    #
    # Linear CKA:
    # M2--M3 = 0.943
    #
    # Therefore:
    # selected_pair = (1, 2)
    # remaining modality = M1
    # --------------------------------------------------------

    model = CALCAM(
        input_dims=input_dims,
        selected_pair=(1, 2),
        d_model=64,
        nhead=4,
        num_shared_layers=1,
        dim_feedforward=256,
        dropout=0.1,
        num_classes=1,
    ).to(device)


    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )


    max_epochs = 30
    patience = 8

    best_macro_f1 = -1.0
    best_state = None
    wait = 0


    for epoch in range(
        1,
        max_epochs + 1,
    ):

        model.train()

        for z1, z2, z3, labels in train_loader:

            z1 = z1.to(device)
            z2 = z2.to(device)
            z3 = z3.to(device)

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
                    z1,
                    z2,
                    z3,
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


        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )


        print(
            f"[seed={seed}] "
            f"epoch={epoch:02d} | "
            f"val acc={val_metrics['accuracy']:.3f} | "
            f"val macro-F1={val_metrics['macro_f1']:.3f}"
        )


        current_score = (
            val_metrics["macro_f1"]
        )


        if (
            current_score
            > best_macro_f1 + 1e-6
        ):

            best_macro_f1 = (
                current_score
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            wait = 0

        else:

            wait += 1

            if wait >= patience:

                print(
                    f"[seed={seed}] "
                    "early stopping"
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
        val_loader,
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
        "CALCAM - C-MAPSS"
    )

    print(
        "VALIDATION RESULTS: mean ± std"
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
            "Directory containing "
            "the C-MAPSS modality embeddings."
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


    (
        z1_train,
        z2_train,
        z3_train,
        y_train,

        z1_val,
        z2_val,
        z3_val,
        y_val,

    ) = load_cmapss(
        args.data_dir
    )


    print(
        "Train:",
        z1_train.shape,
        z2_train.shape,
        z3_train.shape,
        y_train.shape,
    )


    print(
        "Validation:",
        z1_val.shape,
        z2_val.shape,
        z3_val.shape,
        y_val.shape,
    )


    # Class imbalance
    positives = float(
        y_train.sum()
    )

    negatives = float(
        len(y_train)
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
        z1_train,
        z2_train,
        z3_train,
        y_train,
        batch_size=128,
        shuffle=True,
    )


    val_loader = create_loader(
        z1_val,
        z2_val,
        z3_val,
        y_val,
        batch_size=256,
        shuffle=False,
    )


    input_dims = [
        z1_train.shape[1],
        z2_train.shape[1],
        z3_train.shape[1],
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
            val_loader=val_loader,
            input_dims=input_dims,
            device=device,
            positive_weight=positive_weight,
        )

        results.append(
            metrics
        )


    summarize(
        results
    )


if __name__ == "__main__":
    main()
