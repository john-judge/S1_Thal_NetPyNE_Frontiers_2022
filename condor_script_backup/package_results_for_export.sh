#!/bin/bash

BASE_PATH="/staging/j/jjudge3/"

# Check if the user provided an argument
#if [ -z "$1" ]; then
#	    echo "Usage: $0 <directory-name>"
#	        exit 1
#fi

# print the value of $1 for debugging
echo "Argument provided: $1"

# Assign the first argument to a variable
TARGET_DIR="$BASE_PATH/$1"

# Create the directory if it doesn't exist
if [ ! -d "$TARGET_DIR" ]; then
	    mkdir -p "$TARGET_DIR"
	        echo "Created directory: $TARGET_DIR"
	else
		    echo "Directory already exists: $TARGET_DIR"
fi

mv "$BASE_PATH"/* "$TARGET_DIR" 
mv "$BASE_PATH"/output_dir* "$TARGET_DIR" 
mv "$BASE_PATH"/S1_results.tar.gz "$TARGET_DIR" 

# Confirm the move
echo "Moved files (if any) from $BASE_PATH to $TARGET_DIR, ready for export via scp"

# compress the output directory into a single tar.gz file for easier transfer
# do not include the full path in the tar.gz, just the contents of the target dir
tar -czvf "$TARGET_DIR".tar.gz -C "$TARGET_DIR" .
echo "Compressed $TARGET_DIR into $TARGET_DIR.tar.gz for export"

# remove the original target dir after compression
rm -rf "$TARGET_DIR"



