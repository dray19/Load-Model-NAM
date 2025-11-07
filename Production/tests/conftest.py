import sys
from pathlib import Path

# Insert repo root (one level above tests/) at start of sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))