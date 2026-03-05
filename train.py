import os
import pickle
import random
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from model import OCRModel


# ---- Compatibility stub for loading the pickle ----
class ProjectDataset:
    """
    Placeholder class so pickle can load the saved ProjectDataset object.
    The pickle contains attributes like: data, labels, label_mapping, type_mapping.
    """
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state)


# ---- Torch Dataset wrapper ----
class OCRInsuranceTorchDataset(Dataset):
    def __init__(self, data_list, labels_list):
        self.data_list = data_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_tensor, type_tensor = self.data_list[idx]  # (image, type_onehot)
        label = self.labels_list[idx]                    # 0/1
        # Ensure tensors are float32 for model, and label float32 for BCE
        image_tensor = image_tensor.float()
        type_tensor = type_tensor.float()
        y = torch.tensor([label], dtype=torch.float32)   # shape (1,)
        return image_tensor, type_tensor, y


@dataclass
class TrainConfig:
    dataset_path: str = "ocr_insurance_dataset.pkl"
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-3
    seed: int = 42
    train_split: float = 0.7
    val_split: float = 0.15
    # test_split is remainder
    output_dir: str = "outputs"
    model_filename: str = "ocr_model.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pickled_dataset(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def split_indices(n: int, train_split: float, val_split: float, seed: int):
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)

    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)

    train_idx = idxs[:train_end]
    val_idx = idxs[train_end:val_end]
    test_idx = idxs[val_end:]
    return train_idx, val_idx, test_idx


def make_subset(data_list, labels_list, indices):
    return [data_list[i] for i in indices], [labels_list[i] for i in indices]


def accuracy_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    logits: (B,1), y_true: (B,1) float in {0,1}
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == y_true).float().mean().item()
    return correct


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    if not os.path.exists(cfg.dataset_path):
        raise FileNotFoundError(
            f"Dataset file not found: {cfg.dataset_path}\n"
            f"Place it in the project root (same folder as train.py)."
        )

    raw = load_pickled_dataset(cfg.dataset_path)

    data_list = raw.data
    labels_list = raw.labels

    n = len(labels_list)
    train_idx, val_idx, test_idx = split_indices(n, cfg.train_split, cfg.val_split, cfg.seed)

    train_data, train_labels = make_subset(data_list, labels_list, train_idx)
    val_data, val_labels = make_subset(data_list, labels_list, val_idx)

    train_ds = OCRInsuranceTorchDataset(train_data, train_labels)
    val_ds = OCRInsuranceTorchDataset(val_data, val_labels)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCRModel(type_dim=5).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    os.makedirs(cfg.output_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for img, tvec, y in train_loader:
            img, tvec, y = img.to(device), tvec.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(img, tvec)          # (B,1)
            loss = criterion(logits, y)        # y is (B,1)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy_from_logits(logits.detach(), y.detach())

        train_loss /= max(1, len(train_loader))
        train_acc /= max(1, len(train_loader))

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for img, tvec, y in val_loader:
                img, tvec, y = img.to(device), tvec.to(device), y.to(device)
                logits = model(img, tvec)
                loss = criterion(logits, y)

                val_loss += loss.item()
                val_acc += accuracy_from_logits(logits, y)

        val_loss /= max(1, len(val_loader))
        val_acc /= max(1, len(val_loader))

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

    # Save model weights (generated file; not meant to be committed to GitHub)
    model_path = os.path.join(cfg.output_dir, cfg.model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    # Print split info
    print(f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    print("Next: run `python evaluate.py` to generate predictions.csv for the test split.")


if __name__ == "__main__":
    main()
