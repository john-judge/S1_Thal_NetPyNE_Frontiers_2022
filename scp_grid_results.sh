#!/bin/bash

REMOTE_HOST="jjudge3@ap2002.chtc.wisc.edu:/staging/jjudge3/"

# Check that a remote path was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <target ID to download grid.tar.gz to>"
    exit 1
fi

REMOTE_PATH="$REMOTE_HOST/grid*.tar.gz"
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
echo "Extracting $TARGET_DIR/grid*.tar.gz..."
cd "$TARGET_DIR" || {
    echo "Error: Could not change to target directory $TARGET_DIR"
    exit 1
}
TAR_PATTERN="grid*.tar.gz"

# Loop through all files matching the pattern
for archive_file in $TAR_PATTERN; do
  if [ -f "$archive_file" ]; then # Check if it's a regular file
    echo "Extracting: $archive_file"
    tar -xzvf "$archive_file"
  else
    echo "No files found matching pattern: $TAR_PATTERN"
    break # Exit the loop if no files are found
  fi
done

echo "Done. Extracted to $(pwd)/$1"
