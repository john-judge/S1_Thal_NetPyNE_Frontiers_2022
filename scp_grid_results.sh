#!/bin/bash

REMOTE_HOST="jjudge3@ap2002.chtc.wisc.edu:/staging/jjudge3/"

# Check that a remote path was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <target ID to download tune.tar.gz to>"
    exit 1
fi

REMOTE_PATH="$REMOTE_HOST/tune.tar.gz"
TARGET_DIR="C:/Users/jjudge3/Desktop/scratch/S1_Thal_NetPyNE_Frontiers_2022/grid/grid_run$1/"

# Ensure TARGET_DIR exists
mkdir -p "$TARGET_DIR" || {
    echo "Error: Could not create target directory $TARGET_DIR"
    exit 1
}

# Use scp to download the file
echo "Fetching $REMOTE_PATH..."
scp "$REMOTE_PATH" "$TARGET_DIR/" || {
    echo "Error: Failed to fetch file via scp."
    exit 1
}

# Extract the tar.gz archive
echo "Extracting $TARGET_DIR/tune.tar.gz..."
cd "$TARGET_DIR" || {
    echo "Error: Could not change to target directory $TARGET_DIR"
    exit 1
}
tar -xzvf "tune.tar.gz" || {
    echo "Error: Failed to extract archive."
    exit 1
}

echo "Done. Extracted to $(pwd)/$1"
