
#!/bin/bash

# Check if two command-line arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <filename> <word> <control>"
    exit 1
fi

filename="$1"
word_to_search="$2"
tipo_de_busqueda="$3" #0 o 1

# Use grep to search for the word in the file and print the following word
#grep -w  "$word_to_search" "$filename"  | awk '{print $1}'
#printf"$tipo_de_busqueda"
if (( $tipo_de_busqueda == 0)); then
    grep -w  "$word_to_search" "$filename"  | awk '{print $1}'
fi

if (( $tipo_de_busqueda == 1)); then
    grep -w  "$word_to_search" "$filename"  | awk '{print $3}'
fi
