#!/usr/bin/env python3
"""
Run the FLUX.1-Kontext API server
"""

import sys
import os
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.server import main

if __name__ == "__main__":
    main()