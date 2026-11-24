"""
Train and evaluate CALCAM on the Fracture dataset.

Expected local files
--------------------
Zimg_train.npy
Zimg_val.npy
Zimg_test.npy

Ztxt_train.npy
Ztxt_val.npy
Ztxt_test.npy

Ztab_train.npy
Ztab_val.npy
Ztab_test.npy

y_train.npy
y_val.npy
y_test.npy
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
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import DataLoader, TensorDataset

from src.calcam import CALCAM


# ============================================================
# Reproducibility
# ============================================================

SEEDS = [0, 1, 2]


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
        allow_pickle=True
    )


def load_fracture(data_dir):

    # Image
    img_train = load_array(data_dir, "Zimg_train.npy")
    img_val = load_array(data_dir, "Zimg_val.npy")
    img_test = load_array(data_dir, "Zimg_test.npy")

    # Text
    txt_train = load_array(data_dir, "Ztxt_train.npy")
    txt_val = load_array(data_dir, "Ztxt_val.npy")
    txt_test = load_array(data_dir, "Ztxt_test.npy")

    # Tabular
    tab_train = load_array(data_dir, "Ztab_train.npy")
    tab_val = load_array(data_dir, "Ztab_val.npy")
    tab_test = load_array(data_dir, "Ztab_test.npy")

    # Labels
    y_train = load_array(data_dir, "y_train.npy")
    y_val = load_array(data_dir, "y_val.npy")
    y_test = load_array(data_dir, "y_test.npy")

    # Encode labels consistently
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)
    y_val = encoder.transform(y_val)
    y_test = encoder.transform(y_test)

    return (
        img_train.astype(np.float32),
        txt_train.astype(np.float32),
        tab_train.astype(np.float32),
        y_train,

        img_val.astype(np.float32),
        txt_val.astype(np.float32),
        tab_val.astype(np.float32),
        y_val,

        img_test.astype(np.float32),
        txt_test.astype(np.float32),
        tab_test.astype(np.float32),
        y_test,

        len(encoder.classes_),
    )


# ============================================================
# DataLoader
# ============================================================

def create_loader(
    image,
    text,
    tabular,
    labels,
    batch_size,
    shuffle,
):

    dataset = TensorDataset(
        torch.from_numpy(image),
        torch.from_numpy(text),
        torch.from_numpy(tabular),
        torch.tensor(labels, dtype=torch.long),
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

    all_true = []
    all_pred = []
    all_prob = []

    for image, text, tabular, labels in loader:

        image = image.to(device)
        text = text.to(device)
        tabular = tabular.to(device)

        logits = model(
            [
                image,
                text,
                tabular,
            ]
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = probabilities.argmax(
            dim=1
        )

        all_true.append(
            labels.numpy()
        )

        all_pred.append(
            predictions.cpu().numpy()
        )

        all_prob.append(
            probabilities.cpu().numpy()
        )

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "micro_f1": f1_score(
            y_true,
            y_pred,
            average="micro"
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro"
        ),

        "log_loss": log_loss(
            y_true,
            y_prob,
            labels=np.arange(
                y_prob.shape[1]
            )
        ),
    }


# ============================================================
# Training
# ============================================================

def train_one_seed(
    seed,
    train_loader,
    val_loader,
    test_loader,
    input_dims,
    num_classes,
    device,
):

    set_seed(seed)

    # Linear CKA selected:
    # Image -- Tabular
    #
    # modality 0 = Image
    # modality 1 = Text
    # modality 2 = Tabular

    model = CALCAM(
        input_dims=input_dims,
        selected_pair=(0, 2),
        d_model=256,
        nhead=8,
        num_shared_layers=1,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    max_epochs = 30
    patience = 8

    best_macro_f1 = -1.0
    best_state = None
    wait = 0

    for epoch in range(
        1,
        max_epochs + 1
    ):

        model.train()

        for (
            image,
            text,
            tabular,
            labels
        ) in train_loader:

            image = image.to(device)
            text = text.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                [
                    image,
                    text,
                    tabular,
                ]
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        val_metrics = evaluate(
            model,
            val_loader,
            device
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
        test_loader,
        device
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
        "CALCAM - FRACTURE"
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
            dtype=float
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
            "the Fracture embeddings."
        )
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
        img_train,
        txt_train,
        tab_train,
        y_train,

        img_val,
        txt_val,
        tab_val,
        y_val,

        img_test,
        txt_test,
        tab_test,
        y_test,

        num_classes,

    ) = load_fracture(
        args.data_dir
    )

    print(
        "Train:",
        img_train.shape,
        txt_train.shape,
        tab_train.shape
    )

    print(
        "Validation:",
        img_val.shape,
        txt_val.shape,
        tab_val.shape
    )

    print(
        "Test:",
        img_test.shape,
        txt_test.shape,
        tab_test.shape
    )

    print(
        "Classes:",
        num_classes
    )

    train_loader = create_loader(
        img_train,
        txt_train,
        tab_train,
        y_train,
        batch_size=64,
        shuffle=True,
    )

    val_loader = create_loader(
        img_val,
        txt_val,
        tab_val,
        y_val,
        batch_size=128,
        shuffle=False,
    )

    test_loader = create_loader(
        img_test,
        txt_test,
        tab_test,
        y_test,
        batch_size=128,
        shuffle=False,
    )

    input_dims = [
        img_train.shape[1],
        txt_train.shape[1],
        tab_train.shape[1],
    ]

    results = []

    for seed in SEEDS:

        metrics = train_one_seed(
            seed=seed,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            input_dims=input_dims,
            num_classes=num_classes,
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
