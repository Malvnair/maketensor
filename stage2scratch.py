"""
stage2scratch.py
Copy raw HDF5 shards from ARC to scratch.
"""

import subprocess
from pathlib import Path


SRC = Path("/arc/home/$USER$/shards_raw")
DEST = Path("/scratch/$USER$/shards_raw")


def main():

    DEST.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["cp", "-r", f"{SRC}/.", str(DEST)],
        check=True
    )

    size = subprocess.check_output(
        ["du", "-sh", str(DEST)],
        text=True
    ).split()[0]

    print(f"Move complete: {size} copied to {DEST}")


if __name__ == "__main__":
    main()
