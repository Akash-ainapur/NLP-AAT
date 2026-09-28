import sys
from pathlib import Path

# Add project root directory to sys.path for Vercel serverless execution
file_path = Path(__file__).resolve()
parent_dir = file_path.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from app.main import app  # noqa: F401
