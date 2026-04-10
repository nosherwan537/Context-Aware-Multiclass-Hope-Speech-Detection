from pathlib import Path
import sys

# Ensure repository root is on PYTHONPATH for tests in local/Colab environments.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
