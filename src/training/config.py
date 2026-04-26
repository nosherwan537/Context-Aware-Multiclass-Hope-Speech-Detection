from dataclasses import dataclass, field
from typing import List


LABEL2ID = {
    "Not Hope": 0,
    "Realistic Hope": 1,
    "Unrealistic Hope": 2,
    "Generalized Hope": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Training data class counts (roman_Ur_train_clean.csv)
TRAIN_CLASS_COUNTS: List[int] = [1697, 329, 733, 1088]  # order matches LABEL2ID


@dataclass
class TrainingConfig:
    model_name: str = "xlm-roberta-base"
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 8
    warmup_steps: int = 500
    seed: int = 42
    num_labels: int = 4
    max_length: int = 128
    focal_gamma: float = 2.0
    oversample_minority: bool = True      # 2x Realistic Hope
    oversample_target_label: int = 1      # Realistic Hope index
    oversample_factor: int = 2
    train_path: str = "data/cleaned_data/roman_Ur_train_clean.csv"
    dev_path: str = "data/cleaned_data/roman_Ur_dev_clean.csv"
    test_path: str = "data/cleaned_data/roman_Ur_test_without_labels_clean.csv"
    text_col: str = "roman_urdu"
    label_col: str = "multiclass"
    results_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
