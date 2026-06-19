#!/bin/bash

################################################################################
# VLR Series Processing Pipeline - DRY RUN MODE
#
# This script shows what WOULD be done without actually:
# - Downloading VODs
# - Running orchestration
# - Removing files
#
# Useful for:
# - Testing the pipeline
# - Verifying VLR scraping works
# - Checking folder structure
# - Estimating processing time
#
# Usage:
#   ./process_vlr_series_dry_run.sh <vlr_url> [output_base_dir]
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

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

log_dry_run() {
    echo -e "${MAGENTA}[DRY RUN]${NC} Would execute: $1"
}

# Check arguments
if [ $# -lt 1 ]; then
    log_error "Usage: $0 <vlr_url> [output_base_dir]"
    log_info "Example: $0 'https://www.vlr.gg/542272/...' ./output"
    exit 1
fi

VLR_URL="$1"
OUTPUT_BASE_DIR="${2:-./series_output}"

echo ""
echo "========================================================================"
log_warning "DRY RUN MODE - No files will be downloaded or processed"
echo "========================================================================"
echo ""
log_info "VLR URL: $VLR_URL"
log_info "Output directory: $OUTPUT_BASE_DIR"
echo ""

################################################################################
# Step 1: Scrape VLR metadata
################################################################################

log_step "Step 1: Scraping VLR.gg metadata"

TEMP_DIR="$OUTPUT_BASE_DIR/.temp"
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
# Step 2: Extract series information
################################################################################

log_step "Step 2: Extracting series information"

TEAM1=$(jq -r '.teams[0]' "$SERIES_METADATA")
TEAM2=$(jq -r '.teams[1]' "$SERIES_METADATA")
NUM_MAPS=$(jq '.maps | length' "$SERIES_METADATA")

SERIES_NAME="${TEAM1}_vs_${TEAM2}"
SERIES_NAME=$(echo "$SERIES_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

SERIES_DIR="$OUTPUT_BASE_DIR/$SERIES_NAME"
mkdir -p "$SERIES_DIR"

log_info "Series: $TEAM1 vs $TEAM2"
log_info "Maps: $NUM_MAPS"
log_info "Series directory: $SERIES_DIR"
echo ""

cp "$SERIES_METADATA" "$SERIES_DIR/series_metadata.json"

################################################################################
# Step 3: Split metadata
################################################################################

log_step "Step 3: Splitting metadata into individual map files"

MAPS_METADATA_DIR="$SERIES_DIR/metadata"
mkdir -p "$MAPS_METADATA_DIR"

log_info "Splitting metadata..."
valoscribe split-metadata "$SERIES_METADATA" -o "$MAPS_METADATA_DIR" -p "map"

log_success "Metadata split into $NUM_MAPS map files"
echo ""

################################################################################
# Step 4: Show what would be processed
################################################################################

log_step "Step 4: Processing plan for each map"

TOTAL_DURATION=0

for ((i=1; i<=NUM_MAPS; i++)); do
    echo ""
    echo "========================================================================"
    log_step "Map $i/$NUM_MAPS"
    echo "========================================================================"

    MAP_METADATA="$MAPS_METADATA_DIR/map${i}.json"

    if [ ! -f "$MAP_METADATA" ]; then
        log_warning "Map $i metadata not found"
        continue
    fi

    MAP_NAME=$(jq -r '.map' "$MAP_METADATA")
    VOD_URL=$(jq -r '.vod_url' "$MAP_METADATA")

    log_info "Map: $MAP_NAME"
    log_info "VOD URL: $VOD_URL"

    if [ "$VOD_URL" = "null" ] || [ -z "$VOD_URL" ]; then
        log_warning "No VOD URL found for map $i"
        continue
    fi

    MAP_FOLDER_NAME="map${i}_$(echo "$MAP_NAME" | tr '[:upper:]' '[:lower:]')"
    MAP_DIR="$SERIES_DIR/$MAP_FOLDER_NAME"
    mkdir -p "$MAP_DIR"
    mkdir -p "$MAP_DIR/output"

    log_info "Map directory: $MAP_DIR"
    echo ""

    # Show what would be executed
    log_dry_run "valoscribe download \"$VOD_URL\" -o ./temp/videos --height 1080 --fps 60"
    log_info "  → Would download VOD to temporary directory"
    echo ""

    log_dry_run "valoscribe orchestrate process-vod <vod_file> \"$MAP_METADATA\" --output \"$MAP_DIR/output\" --fps 4 --quiet"
    log_info "  → Would process VOD and extract events"
    log_info "  → Output: frame_states.csv, event_log.jsonl"
    echo ""

    log_dry_run "rm -f <vod_file>"
    log_info "  → Would remove VOD to save disk space"
    echo ""

    # Try to estimate duration (this is a rough estimate)
    # Typical Valorant match is 30-60 minutes
    ESTIMATED_DURATION=35
    TOTAL_DURATION=$((TOTAL_DURATION + ESTIMATED_DURATION))

    log_info "Estimated processing time: ~${ESTIMATED_DURATION} minutes"
    echo ""
done

################################################################################
# Summary
################################################################################

echo ""
echo "========================================================================"
log_success "Dry run completed!"
echo "========================================================================"
echo ""
log_info "Summary:"
log_info "  Series: $TEAM1 vs $TEAM2"
log_info "  Total maps: $NUM_MAPS"
log_info "  Estimated total processing time: ~${TOTAL_DURATION} minutes"
echo ""
log_info "Output structure (created):"
tree -L 3 "$SERIES_DIR" 2>/dev/null || find "$SERIES_DIR" -type d | sed 's|[^/]*/| |g'
echo ""
log_info "To actually process this series, run:"
echo "  ./scripts/process_vlr_series.sh \"$VLR_URL\" \"$OUTPUT_BASE_DIR\""
echo ""
log_warning "Dry run artifacts (metadata files) have been created but no VODs were downloaded"
echo ""
