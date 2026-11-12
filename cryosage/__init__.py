"""CryoSAGE reconstruction package."""
import os
from pathlib import Path
import tempfile

__version__ = "0.1.0"

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"cryosage-matplotlib"))
