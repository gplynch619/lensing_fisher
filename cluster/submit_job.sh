#!/bin/bash

# job_submit.sh - Creates and submits a SLURM job for a specific input file
# Usage: ./job_submit.sh -f /path/to/file -t /path/to/template

# Default values
INPUT_FILE=""
TEMPLATE_FILE="./job_template.sh"

# Parse command line arguments
while getopts ":f:t:" opt; do
  case $opt in
    f) INPUT_FILE="$OPTARG" ;;
    t) TEMPLATE_FILE="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

# Check if required arguments are provided
if [ -z "$INPUT_FILE" ]; then
    echo "Usage: $0 -f /path/to/file [-t /path/to/template]"
    exit 1
fi

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Check if the template file exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: Template file '$TEMPLATE_FILE' not found"
    exit 1
fi

# Get the absolute path of the input file
INPUT_FILE=$(realpath "$INPUT_FILE")

# Get the filename without path
FILENAME=$(basename "$INPUT_FILE")

# Create a job name based on the filename (removing extensions)
JOB_NAME=$(echo "$FILENAME" | sed 's/\.[^.]*$//')

# Create a temporary submission script
SUB_DIR="submission_scripts"
mkdir -p "$SUB_DIR"

TEMP_SCRIPT="submission_scripts/submit_${JOB_NAME}.sh"

# Copy the template and replace placeholders
cat "$TEMPLATE_FILE" > "$TEMP_SCRIPT"

# Replace template variables with actual values
sed -i "s|{{JOB_NAME}}|${JOB_NAME}|g" "$TEMP_SCRIPT"
sed -i "s|{{INPUT_FILE}}|${INPUT_FILE}|g" "$TEMP_SCRIPT"

# Make the script executable
chmod +x "$TEMP_SCRIPT"

# Submit the job
echo "Submitting job for $FILENAME"
sbatch "$TEMP_SCRIPT"

# Return success
exit 0