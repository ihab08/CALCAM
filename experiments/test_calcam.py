import torch

from src.calcam import CALCAM


def main():
    batch_size = 8

    # Example dimensions matching Fracture embeddings.
    image = torch.randn(batch_size, 1024)
    text = torch.randn(batch_size, 896)
    tabular = torch.randn(batch_size, 128)

    # Linear CKA selects Image--Tabular:
    # Image = index 0
    # Text = index 1
    # Tabular = index 2
    model = CALCAM(
        input_dims=[1024, 896, 128],
        selected_pair=(0, 2),
        d_model=256,
        nhead=8,
        num_shared_layers=1,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=10,
    )

    logits, attention = model(
        [image, text, tabular],
        return_attention=True,
    )

    print("Logits shape:", logits.shape)

    assert logits.shape == (
        batch_size,
        10,
    )

    print(
        "Selected pair:",
        model.selected_pair,
    )

    print(
        "Remaining modality:",
        model.remaining_index,
    )

    print(
        "CALCAM forward test passed."
    )


if __name__ == "__main__":
    main()
