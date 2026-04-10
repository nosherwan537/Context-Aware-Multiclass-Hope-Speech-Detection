from dataclasses import dataclass


@dataclass
class TrainingConfig:
    model_name: str = "xlm-roberta-base"
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 5
    warmup_steps: int = 500
    seed: int = 42
    num_labels: int = 4
