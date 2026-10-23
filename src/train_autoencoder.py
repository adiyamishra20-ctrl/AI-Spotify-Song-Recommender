import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SongAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, features):
        return self.decoder(self.encoder(features))


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    models = root / "models"
    reports = root / "reports"

    models.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    songs = pd.read_csv(processed / "baseline_songs.csv")
    features = pd.read_csv(processed / "baseline_features.csv")

    if (
        songs["row_id"].duplicated().any()
        or features["row_id"].duplicated().any()
        or not songs["row_id"].equals(features["row_id"])
    ):
        raise ValueError("Song and feature IDs do not align.")

    feature_columns = features.columns.drop("row_id").tolist()
    values = features[feature_columns].to_numpy(dtype=np.float32)

    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("Expected finite binary features.")

    # Keep identical feature vectors in the same split.
    # Otherwise identical examples could appear in training and validation.
    groups = [
        hashlib.sha256(row.tobytes()).hexdigest()
        for row in values
    ]

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )
    train_indices, validation_indices = next(
        splitter.split(values, groups=groups)
    )

    tensor = torch.from_numpy(values)
    train_tensor = tensor[train_indices]
    validation_tensor = tensor[validation_indices]

    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    model = SongAutoencoder(input_dim=values.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.0001
    )

    # Inputs are binary; the decoder outputs logits for this loss.
    loss_function = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []

    print(f"Training songs: {len(train_indices)}")
    print(f"Validation songs: {len(validation_indices)}")
    print(f"Features: {values.shape[1]}; embedding size: 16")

    # Compare reconstruction against a simple constant prediction.
    probabilities = train_tensor.mean(dim=0).clamp(0.0001, 0.9999)
    constant_logits = torch.logit(probabilities).expand_as(
        validation_tensor
    )
    constant_loss = loss_function(
        constant_logits, validation_tensor
    ).item()

    for epoch in range(1, 301):
        model.train()
        total_loss = 0.0

        for (batch,) in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch), batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        training_loss = total_loss / len(train_tensor)

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(validation_tensor), validation_tensor
            ).item()

        if not np.isfinite([training_loss, validation_loss]).all():
            raise RuntimeError("Training produced a non-finite loss.")

        history.append({
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
        })

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:3d} | train {training_loss:.4f} "
                f"| validation {validation_loss:.4f}",
                flush=True,
            )

        if validation_loss < best_loss - 0.00001:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= 25:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    model.eval()

    # Validation songs are encoded, but were not used for gradient updates.
    with torch.no_grad():
        embeddings = model.encoder(tensor).numpy()

    embedding_table = pd.DataFrame(
        embeddings,
        columns=[f"embedding_{i}" for i in range(16)],
    )
    embedding_table.insert(0, "row_id", songs["row_id"])
    embedding_table.to_csv(
        processed / "song_embeddings.csv", index=False
    )

    split_table = songs[["row_id"]].copy()
    split_table["split"] = "train"
    split_table.loc[validation_indices, "split"] = "validation"
    split_table.to_csv(reports / "autoencoder_split.csv", index=False)

    torch.save(
        {
            "state_dict": best_state,
            "input_dim": values.shape[1],
            "embedding_dim": 16,
            "feature_columns": feature_columns,
        },
        models / "autoencoder.pt",
    )

    pd.DataFrame(history).to_csv(
        reports / "autoencoder_training.csv", index=False
    )

    metrics = {
        "seed": 42,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "constant_validation_loss": constant_loss,
        "training_songs": len(train_indices),
        "validation_songs": len(validation_indices),
    }
    (reports / "autoencoder_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Constant-prediction loss: {constant_loss:.4f}")
    print(f"Embedding shape: {embeddings.shape}")
    print("Saved model, embeddings, split, and training history.")


if __name__ == "__main__":
    main()