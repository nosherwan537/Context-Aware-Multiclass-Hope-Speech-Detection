import random
import numpy as np
import torch

from .config import TrainingConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    cfg = TrainingConfig()
    set_seed(cfg.seed)
    print("[Baseline] Training scaffold ready.")
    print("Expected data files: data/splits/train.csv, val.csv, test.csv")
    print(f"Model={cfg.model_name} bs={cfg.batch_size} lr={cfg.learning_rate} epochs={cfg.epochs}")


if __name__ == "__main__":
    main()
