from src.normalizer import PhoneticNormalizer
from src.model import DualHeadXLMRClassifier
from src.model.dual_head_xlmr import DualHeadConfig


def main() -> None:
    _ = PhoneticNormalizer()
    _ = DualHeadXLMRClassifier(DualHeadConfig())
    print("[HLSP] Training scaffold ready (normalizer + dual-head model initialized).")


if __name__ == "__main__":
    main()
