#!/bin/bash

################################################################################
# VLR Series Processing Pipeline
#
# Automates the full pipeline for processing Valorant matches from VLR.gg:
# 1. Scrapes VLR.gg for match metadata
# 2. Splits metadata into individual map files
# 3. For each map:
#    - Downloads YouTube VOD
#    - Runs orchestration to extract events
#    - Saves output in organized folder structure
#    - Removes VOD to save space
#
# Usage:
#   ./process_vlr_series.sh <vlr_url> [output_base_dir]
#
# Example:
#   ./process_vlr_series.sh "https://www.vlr.gg/542272/..." ./output
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory (for accessing config files)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Print colored message
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Check arguments
if [ $# -lt 1 ]; then
    log_error "Usage: $0 <vlr_url> [output_base_dir]"
    log_info "Example: $0 'https://www.vlr.gg/542272/...' ./output"
    exit 1
fi

VLR_URL="$1"
OUTPUT_BASE_DIR="${2:-./series_output}"

# Extract match ID from URL for unique directory naming
# Format: https://www.vlr.gg/542265/team1-vs-team2-event
MATCH_ID=$(echo "$VLR_URL" | grep -oE '/[0-9]+/' | head -1 | tr -d '/')

if [ -z "$MATCH_ID" ]; then
    log_error "Could not extract match ID from URL: $VLR_URL"
    log_error "Expected format: https://www.vlr.gg/MATCH_ID/..."
    exit 1
fi

log_info "Starting VLR series processing pipeline"
log_info "VLR URL: $VLR_URL"
log_info "Match ID: $MATCH_ID"
log_info "Output directory: $OUTPUT_BASE_DIR"
echo ""

################################################################################
# Step 1: Scrape VLR metadata
################################################################################

log_step "Step 1: Scraping VLR.gg metadata"

# Create temp directory for intermediate files (unique per match for parallel processing)
TEMP_DIR="$OUTPUT_BASE_DIR/.temp_${MATCH_ID}"
mkdir -p "$TEMP_DIR"

SERIES_METADATA="$TEMP_DIR/series_metadata.json"

log_info "Scraping match metadata..."
valoscribe scrape-vlr "$VLR_URL" -o "$SERIES_METADATA"

if [ ! -f "$SERIES_METADATA" ]; then
    log_error "Failed to scrape VLR metadata"
    exit 1
fi

log_success "Metadata scraped successfully"
echo ""

################################################################################
# Step 2: Extract series information and create organized folder structure
################################################################################

log_step "Step 2: Extracting series information"

# Extract team names and create series folder
TEAM1=$(jq -r '.teams[0]' "$SERIES_METADATA")
TEAM2=$(jq -r '.teams[1]' "$SERIES_METADATA")
NUM_MAPS=$(jq '.maps | length' "$SERIES_METADATA")

# Create series folder name (include match ID for uniqueness)
# Format: <match_id>_<team1>_vs_<team2>
# Example: 542265_paper_rex_vs_g2_esports
SERIES_NAME="${MATCH_ID}_${TEAM1}_vs_${TEAM2}"
SERIES_NAME=$(echo "$SERIES_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

SERIES_DIR="$OUTPUT_BASE_DIR/$SERIES_NAME"
mkdir -p "$SERIES_DIR"

log_info "Series: $TEAM1 vs $TEAM2 (Match ID: $MATCH_ID)"
log_info "Maps: $NUM_MAPS"
log_info "Series directory: $SERIES_DIR"
echo ""

# Copy series metadata to series folder
cp "$SERIES_METADATA" "$SERIES_DIR/series_metadata.json"

################################################################################
# Step 3: Split metadata into individual map files
################################################################################

log_step "Step 3: Splitting metadata into individual map files"

MAPS_METADATA_DIR="$SERIES_DIR/metadata"
mkdir -p "$MAPS_METADATA_DIR"

log_info "Splitting metadata..."
valoscribe split-metadata "$SERIES_METADATA" -o "$MAPS_METADATA_DIR" -p "map"

log_success "Metadata split into $NUM_MAPS map files"
echo ""

################################################################################
# Step 4: Process each map
################################################################################

log_step "Step 4: Processing individual maps"

# Process each map
for ((i=1; i<=NUM_MAPS; i++)); do
    echo ""
    echo "========================================================================"
    log_step "Processing Map $i/$NUM_MAPS"
    echo "========================================================================"

    MAP_METADATA="$MAPS_METADATA_DIR/map${i}.json"

    if [ ! -f "$MAP_METADATA" ]; then
        log_warning "Map $i metadata not found, skipping..."
        continue
    fi

    # Extract map information
    MAP_NAME=$(jq -r '.map' "$MAP_METADATA")
    VOD_URL=$(jq -r '.vod_url' "$MAP_METADATA")

    log_info "Map: $MAP_NAME"
    log_info "VOD URL: $VOD_URL"

    # Check if VOD URL exists
    if [ "$VOD_URL" = "null" ] || [ -z "$VOD_URL" ]; then
        log_warning "No VOD URL found for map $i, skipping..."
        continue
    fi

    # Create map folder (e.g., "map1_haven")
    MAP_FOLDER_NAME="map${i}_$(echo "$MAP_NAME" | tr '[:upper:]' '[:lower:]')"
    MAP_DIR="$SERIES_DIR/$MAP_FOLDER_NAME"
    mkdir -p "$MAP_DIR"

    log_info "Map directory: $MAP_DIR"

    # Define paths
    VOD_DOWNLOAD_DIR="$TEMP_DIR/videos"
    mkdir -p "$VOD_DOWNLOAD_DIR"

    OUTPUT_DIR="$MAP_DIR/output"
    mkdir -p "$OUTPUT_DIR"

    # Create log file for this map
    LOG_FILE="$OUTPUT_DIR/processing.log"

    #---------------------------------------------------------------------------
    # Check if output already exists
    #---------------------------------------------------------------------------

    FRAME_STATES_FILE="$OUTPUT_DIR/frame_states.csv"
    EVENT_LOG_FILE="$OUTPUT_DIR/event_log.jsonl"

    if [ -f "$FRAME_STATES_FILE" ] && [ -f "$EVENT_LOG_FILE" ]; then
        log_success "[Map $i] Output already exists, skipping processing"
        log_info "  - $FRAME_STATES_FILE"
        log_info "  - $EVENT_LOG_FILE"
        log_info "  - $LOG_FILE (existing log)"
        continue
    fi

    # Start logging for this map
    # All output below will be captured to log file AND displayed
    exec 3>&1 4>&2  # Save original stdout/stderr
    exec > >(tee -a "$LOG_FILE") 2>&1  # Redirect to both file and console

    echo "========================================================================="
    echo "Map $i/$NUM_MAPS Processing Log"
    echo "========================================================================="
    echo "Map: $MAP_NAME"
    echo "VOD URL: $VOD_URL"
    echo "Started: $(date)"
    echo "========================================================================="
    echo ""

    #---------------------------------------------------------------------------
    # Step 4a: Download YouTube VOD
    #---------------------------------------------------------------------------

    log_info "[Map $i] Downloading YouTube VOD..."

    # Check for optional start_time and duration in metadata (for livestream clips)
    START_TIME=$(jq -r '.start_time // empty' "$MAP_METADATA" 2>/dev/null || echo "")
    DURATION=$(jq -r '.duration // empty' "$MAP_METADATA" 2>/dev/null || echo "")

    # Build download command with optional timestamp parameters
    DOWNLOAD_CMD="valoscribe download \"$VOD_URL\" -o \"$VOD_DOWNLOAD_DIR\" --height 1080 --fps 60"
    if [ -n "$START_TIME" ]; then
        DOWNLOAD_CMD="$DOWNLOAD_CMD --start $START_TIME"
        log_info "  Start time: ${START_TIME}s (from metadata)"
    fi
    if [ -n "$DURATION" ]; then
        DOWNLOAD_CMD="$DOWNLOAD_CMD --duration $DURATION"
        log_info "  Duration: ${DURATION}s (from metadata)"
    fi

    # Execute download
    eval $DOWNLOAD_CMD

    # Find the downloaded video file (most recent .mp4 in download dir)
    # Use ls -t which works on both macOS and Linux
    VOD_FILE=$(ls -t "$VOD_DOWNLOAD_DIR"/*.mp4 2>/dev/null | head -1)

    if [ -z "$VOD_FILE" ] || [ ! -f "$VOD_FILE" ]; then
        log_error "[Map $i] Failed to download VOD"
        continue
    fi

    log_success "[Map $i] VOD downloaded: $(basename "$VOD_FILE")"

    #---------------------------------------------------------------------------
    # Step 4b: Run orchestration
    #---------------------------------------------------------------------------

    log_info "[Map $i] Running orchestration (this may take a while)..."

    # Run orchestration with quiet mode (only show events)
    valoscribe orchestrate process-vod "$VOD_FILE" "$MAP_METADATA" \
        --output "$OUTPUT_DIR" \
        --fps 4 \
        --quiet \
        --mute-agent-detector
        # --show

    if [ $? -eq 0 ]; then
        log_success "[Map $i] Orchestration completed successfully"
        log_info "[Map $i] Output files:"
        log_info "  - $OUTPUT_DIR/frame_states.csv"
        log_info "  - $OUTPUT_DIR/event_log.jsonl"
    else
        log_error "[Map $i] Orchestration failed"
    fi

    #---------------------------------------------------------------------------
    # Step 4c: Copy metadata to map folder
    #---------------------------------------------------------------------------

    cp "$MAP_METADATA" "$MAP_DIR/metadata.json"

    #---------------------------------------------------------------------------
    # Step 4d: Remove VOD to save space
    #---------------------------------------------------------------------------

    log_info "[Map $i] Removing VOD to save space..."
    rm -f "$VOD_FILE"
    log_success "[Map $i] VOD removed"

    # Close log file and restore stdout/stderr
    echo ""
    echo "========================================================================="
    echo "Map $i Processing Complete"
    echo "Finished: $(date)"
    echo "========================================================================="

    exec 1>&3 2>&4 3>&- 4>&-  # Restore original stdout/stderr

    # Final summary for console (not in log file)
    log_success "[Map $i] Processing complete, log saved to: $LOG_FILE"

    echo ""
done

################################################################################
# Step 5: Cleanup and summary
################################################################################

log_step "Step 5: Cleanup and summary"

# Remove temp directory
log_info "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo ""
echo "========================================================================"
log_success "Pipeline completed successfully!"
echo "========================================================================"
echo ""
log_info "Series: $TEAM1 vs $TEAM2"
log_info "Maps processed: $NUM_MAPS"
log_info "Output directory: $SERIES_DIR"
echo ""
log_info "Folder structure:"
tree -L 2 "$SERIES_DIR" 2>/dev/null || find "$SERIES_DIR" -maxdepth 2 -type d | sed 's|[^/]*/| |g'
echo ""
log_info "Next steps:"
log_info "  - Review event logs: $SERIES_DIR/map*/output/event_log.jsonl"
log_info "  - Analyze frame states: $SERIES_DIR/map*/output/frame_states.csv"
log_info "  - Series metadata: $SERIES_DIR/series_metadata.json"
echo ""
