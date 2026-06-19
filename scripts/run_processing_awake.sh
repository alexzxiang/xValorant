#!/bin/bash
#
# Run parallel processing with caffeinate to keep laptop awake
#
# This script prevents your MacBook from sleeping during the long processing job.
# Keep the laptop plugged in and the lid open while this runs.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo "Starting Parallel Processing (with caffeinate)"
echo "========================================================================"
echo ""
echo "This will keep your laptop awake until processing completes."
echo ""
echo "Tips:"
echo "  - Keep laptop plugged in"
echo "  - Keep lid open (or connect external display)"
echo "  - Lower screen brightness to save power"
echo ""
echo "Press Ctrl+C to cancel if needed..."
sleep 3

# Run with caffeinate to prevent sleep
# -d: prevent display sleep
# -i: prevent idle sleep
# -m: prevent disk sleep
caffeinate -dim "$SCRIPT_DIR/process_all_series_parallel.sh"

echo ""
echo "========================================================================"
echo "Processing complete - system sleep settings restored"
echo "========================================================================"
