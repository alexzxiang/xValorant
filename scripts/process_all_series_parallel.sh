#!/bin/bash
#
# Parallel Series Processor
#
# Processes multiple VLR match series in parallel for maximum throughput.
# Each series is processed independently, allowing efficient use of multi-core systems.
#

set -e

#==============================================================================
# CONFIGURATION
#==============================================================================

# Number of parallel jobs (recommended: 5 for 14-core MacBook Pro)
# Adjust based on your system:
#   - 8 cores: 3-4 jobs
#   - 14 cores: 5-6 jobs
#   - 16+ cores: 6-8 jobs
PARALLEL_JOBS=7

#==============================================================================
# MATCH URLs FILE
#==============================================================================
# Match URLs are loaded from matches.txt in the same directory
# Format: One URL per line, lines starting with # are ignored
#
MATCHES_FILE="$(dirname "${BASH_SOURCE[0]}")/matches_part1.txt"

#==============================================================================
# SCRIPT SETUP
#==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROCESS_SCRIPT="$SCRIPT_DIR/process_vlr_series.sh"

# Verify process_vlr_series.sh exists
if [ ! -f "$PROCESS_SCRIPT" ]; then
    echo "ERROR: process_vlr_series.sh not found at: $PROCESS_SCRIPT"
    exit 1
fi

# Verify matches.txt exists
if [ ! -f "$MATCHES_FILE" ]; then
    echo "ERROR: matches.txt not found at: $MATCHES_FILE"
    echo ""
    echo "Create matches.txt with one match URL per line:"
    echo "  https://www.vlr.gg/MATCH_ID/team1-vs-team2-event"
    echo ""
    exit 1
fi

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "ERROR: GNU parallel is not installed"
    echo ""
    echo "Install with:"
    echo "  brew install parallel"
    echo ""
    exit 1
fi

# Load match URLs from file (skip comments and empty lines)
SERIES_URLS=()
while IFS= read -r line; do
    SERIES_URLS+=("$line")
done < <(grep -v '^#' "$MATCHES_FILE" | grep -v '^$')

#==============================================================================
# PROCESSING FUNCTION
#==============================================================================

process_series() {
    local URL=$1
    local MATCH_ID=$(echo "$URL" | grep -oE '/[0-9]+/' | head -1 | tr -d '/')

    echo "========================================================================"
    echo "Processing Match ID: $MATCH_ID"
    echo "========================================================================"

    START_TIME=$(date +%s)

    # Run the processor
    "$PROCESS_SCRIPT" "$URL"
    EXIT_CODE=$?

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))

    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Match $MATCH_ID completed successfully in ${MINUTES}m ${SECONDS}s"
    else
        echo "✗ Match $MATCH_ID failed (exit code: $EXIT_CODE) after ${MINUTES}m ${SECONDS}s"
    fi
    echo "========================================================================"
    echo ""

    return $EXIT_CODE
}

# Export function and variables
export -f process_series
export PROCESS_SCRIPT

#==============================================================================
# MAIN EXECUTION
#==============================================================================

TOTAL_SERIES=${#SERIES_URLS[@]}

echo "========================================================================"
echo "Parallel Series Processor"
echo "========================================================================"
echo "Total series to process: $TOTAL_SERIES"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Script: $PROCESS_SCRIPT"
echo ""
echo "Processing will run $PARALLEL_JOBS matches concurrently."
echo "Each line is tagged with [Match ID] for easy tracking."
echo "========================================================================"
echo ""

# Validate we have URLs
if [ $TOTAL_SERIES -eq 0 ]; then
    echo "ERROR: No match URLs configured"
    echo "Please add match URLs to the SERIES_URLS array in this script"
    exit 1
fi

echo "Starting parallel processing..."
echo ""

# Start timestamp
OVERALL_START=$(date +%s)

# Process all series in parallel with organized output
# --jobs: Number of parallel jobs (5)
# --line-buffer: Keep output lines together per job
# --tagstring: Prefix each line with match ID for identification
# --bar: Show progress bar
printf '%s\n' "${SERIES_URLS[@]}" | parallel --jobs $PARALLEL_JOBS \
    --line-buffer \
    --tagstring '[Match {= s:.*?/([0-9]+)/.*:$1:; =}]' \
    --bar \
    process_series

EXIT_CODE=$?

echo ""
echo "All jobs completed."
echo ""

# End timestamp and summary
OVERALL_END=$(date +%s)
TOTAL_DURATION=$((OVERALL_END - OVERALL_START))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "========================================================================"
echo "Processing Complete"
echo "========================================================================"
echo "Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo ""

# Count completed outputs
OUTPUT_BASE="${OUTPUT_BASE:-output}"
if [ -d "$OUTPUT_BASE" ]; then
    COMPLETED_MAPS=$(find "$OUTPUT_BASE" -name "event_log.jsonl" 2>/dev/null | wc -l | tr -d ' ')
    echo "Total maps processed: $COMPLETED_MAPS"
    echo "Output directory: $OUTPUT_BASE"
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ All series processed successfully!"
else
    echo ""
    echo "⚠ Some series failed to process (check logs above)"
fi

echo "========================================================================"

exit $EXIT_CODE
