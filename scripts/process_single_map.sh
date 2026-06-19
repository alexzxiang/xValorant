#!/bin/bash

################################################################################
# Single Map Processing Script
#
# Process a single map when you already have:
# - Downloaded VOD file
# - Map metadata JSON
#
# This is useful when:
# - You want to re-process a map with different settings
# - You've already downloaded VODs separately
# - You want to process just one map from a series
#
# Usage:
#   ./process_single_map.sh <vod_file> <metadata_file> <output_dir> [--keep-vod]
#
# Example:
#   ./process_single_map.sh map1.mp4 map1_metadata.json ./output
#   ./process_single_map.sh map1.mp4 map1_metadata.json ./output --keep-vod
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Check arguments
if [ $# -lt 3 ]; then
    log_error "Usage: $0 <vod_file> <metadata_file> <output_dir> [--keep-vod]"
    log_info "Example: $0 map1.mp4 map1_metadata.json ./output"
    log_info "Example: $0 map1.mp4 map1_metadata.json ./output --keep-vod"
    exit 1
fi

VOD_FILE="$1"
METADATA_FILE="$2"
OUTPUT_DIR="$3"
KEEP_VOD=false

# Check for --keep-vod flag
if [ $# -ge 4 ] && [ "$4" = "--keep-vod" ]; then
    KEEP_VOD=true
fi

# Validate inputs
if [ ! -f "$VOD_FILE" ]; then
    log_error "VOD file not found: $VOD_FILE"
    exit 1
fi

if [ ! -f "$METADATA_FILE" ]; then
    log_error "Metadata file not found: $METADATA_FILE"
    exit 1
fi

# Extract map info from metadata
MAP_NAME=$(jq -r '.map' "$METADATA_FILE" 2>/dev/null || echo "Unknown")
MAP_NUMBER=$(jq -r '.map_number' "$METADATA_FILE" 2>/dev/null || echo "?")

echo ""
log_info "Processing single map"
log_info "VOD: $VOD_FILE"
log_info "Metadata: $METADATA_FILE"
log_info "Map: $MAP_NAME (Map $MAP_NUMBER)"
log_info "Output: $OUTPUT_DIR"
if [ "$KEEP_VOD" = true ]; then
    log_info "Keep VOD: Yes"
else
    log_info "Keep VOD: No (will be deleted after processing)"
fi
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

################################################################################
# Step 1: Run orchestration
################################################################################

log_step "Running orchestration"

log_info "This may take 20-40 minutes depending on VOD length..."
echo ""

valoscribe orchestrate process-vod "$VOD_FILE" "$METADATA_FILE" \
    --output "$OUTPUT_DIR" \
    --fps 4 \
    --quiet \
    --mute-agent-detector

if [ $? -eq 0 ]; then
    log_success "Orchestration completed successfully"
    echo ""
    log_info "Output files:"
    log_info "  - $OUTPUT_DIR/frame_states.csv"
    log_info "  - $OUTPUT_DIR/event_log.jsonl"
else
    log_error "Orchestration failed"
    exit 1
fi

################################################################################
# Step 2: Copy metadata to output
################################################################################

log_step "Copying metadata to output directory"

cp "$METADATA_FILE" "$OUTPUT_DIR/metadata.json"
log_success "Metadata copied"

################################################################################
# Step 3: Generate summary
################################################################################

log_step "Generating summary"

EVENT_LOG="$OUTPUT_DIR/event_log.jsonl"

if [ -f "$EVENT_LOG" ]; then
    TOTAL_EVENTS=$(wc -l < "$EVENT_LOG" | tr -d ' ')
    ROUND_STARTS=$(grep -c '"type":"round_start"' "$EVENT_LOG" || echo "0")
    KILLS=$(grep -c '"type":"kill"' "$EVENT_LOG" || echo "0")
    ABILITIES=$(grep -c '"type":"ability_used"' "$EVENT_LOG" || echo "0")
    ULTIMATES=$(grep -c '"type":"ultimate_used"' "$EVENT_LOG" || echo "0")

    echo ""
    log_info "Event Summary:"
    log_info "  Total events: $TOTAL_EVENTS"
    log_info "  Rounds: $ROUND_STARTS"
    log_info "  Kills: $KILLS"
    log_info "  Abilities used: $ABILITIES"
    log_info "  Ultimates used: $ULTIMATES"
fi

################################################################################
# Step 4: Remove VOD if requested
################################################################################

if [ "$KEEP_VOD" = false ]; then
    log_step "Removing VOD to save space"

    VOD_SIZE=$(du -h "$VOD_FILE" | cut -f1)
    rm -f "$VOD_FILE"

    log_success "Removed VOD (freed $VOD_SIZE)"
fi

################################################################################
# Summary
################################################################################

echo ""
echo "========================================================================"
log_success "Processing complete!"
echo "========================================================================"
echo ""
log_info "Map: $MAP_NAME (Map $MAP_NUMBER)"
log_info "Output directory: $OUTPUT_DIR"
echo ""
log_info "Next steps:"
log_info "  - Review events: $OUTPUT_DIR/event_log.jsonl"
log_info "  - Analyze frames: $OUTPUT_DIR/frame_states.csv"
echo ""
