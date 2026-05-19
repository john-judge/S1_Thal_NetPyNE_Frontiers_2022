#!/bin/bash

REMOTE_HOST="jjudge3@ap2002.chtc.wisc.edu:/staging/j/jjudge3/"

# Check that a remote path was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <dir in jjudge3@ap2002.chtc.wisc.edu:/staging/j/jjudge3/>"
    exit 1
fi

REMOTE_PATH="$REMOTE_HOST/$1/*"
TARGET_DIR="C:/Users/jjudge3/Desktop/neuron docker/composed_results/$1/"

# Ensure TARGET_DIR exists
mkdir -p "$TARGET_DIR" || {
    echo "Error: Could not create target directory $TARGET_DIR"
    exit 1
}

# Use scp to download the file
echo "Fetching $REMOTE_PATH..."
scp "$REMOTE_PATH" "$TARGET_DIR/" || {
    echo "Error: Failed to fetch file via scp, trying .tar.gz extension..."
    REMOTE_PATH="$REMOTE_HOST/$1.tar.gz"
    scp "$REMOTE_PATH" "$TARGET_DIR/" || {
        echo "Error: Failed to fetch file via scp."
        exit 1
    }
}

# if the file is a .tar.gz, extract it
if [[ "$REMOTE_PATH" == *.tar.gz ]]; then
    echo "Extracting $TARGET_DIR/$(basename "$REMOTE_PATH")..."
    tar -xzvf "$TARGET_DIR/$(basename "$REMOTE_PATH")" -C "$TARGET_DIR" || {
        echo "Error: Failed to extract archive."
        exit 1
    }
    # Optionally, remove the .tar.gz file after extraction
    #rm "$TARGET_DIR/$(basename "$REMOTE_PATH")"
fi

# Extract the tar.gz archive inside the target directory if it exists
echo "Extracting $TARGET_DIR/S1_results.tar.gz..."
cd "$TARGET_DIR" || {
    echo "Error: Could not change to target directory $TARGET_DIR"
    exit 1
}
tar -xzvf "S1_results.tar.gz" || {
    echo "Error: Failed to extract archive."
    exit 1
}

echo "Done. Extracted to $(pwd)/$1"