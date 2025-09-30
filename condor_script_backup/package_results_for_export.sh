#!/bin/bash

BASE_PATH="/staging/jjudge3/"

# Check if the user provided an argument
if [ -z "$1" ]; then
	    echo "Usage: $0 <directory-name>"
	        exit 1
fi

# Assign the first argument to a variable
TARGET_DIR="$BASE_PATH/$1"

# Create the directory if it doesn't exist
if [ ! -d "$TARGET_DIR" ]; then
	    mkdir -p "$TARGET_DIR"
	        echo "Created directory: $TARGET_DIR"
	else
		    echo "Directory already exists: $TARGET_DIR"
fi

mv "$BASE_PATH"/output_dir_*.tar.gz "$TARGET_DIR" 
mv "$BASE_PATH"/S1_results.tar.gz "$TARGET_DIR" 

# Confirm the move
echo "Moved files (if any) from $BASE_PATH to $TARGET_DIR, ready for export via scp"



