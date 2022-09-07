import sys

if sys.version_info < (3, 9):
    sys.exit(f"Must be using at least Python 3.9, you are using {sys.version}")
