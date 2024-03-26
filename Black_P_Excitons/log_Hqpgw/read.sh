
#!/bin/bash

# Check if two command-line arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <filename> <word>"
    exit 1
fi

filename="$1"
word_to_search="$2"

# Use grep to search for the word in the file and print the following word
grep -w  "$word_to_search" "$filename"  | awk '{print $3}'

