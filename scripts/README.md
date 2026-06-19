# VLR Series Processing Scripts

Automated scripts for processing Valorant match series from VLR.gg.

## Quick Start

```bash
# Full pipeline: scrape VLR, download VODs, process everything
./scripts/process_vlr_series.sh "https://www.vlr.gg/542272/..." ./output

# Dry run: test without downloading/processing
./scripts/process_vlr_series_dry_run.sh "https://www.vlr.gg/542272/..." ./output

# Process single map (when you already have VOD and metadata)
./scripts/process_single_map.sh map1.mp4 map1_metadata.json ./output
```

## Scripts Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| **process_vlr_series.sh** | Full automated pipeline | Process entire match series from VLR URL |
| **process_vlr_series_dry_run.sh** | Test without processing | Verify VLR scraping and folder structure |
| **process_single_map.sh** | Process one map | Re-process or process pre-downloaded VODs |

---

## process_vlr_series.sh

A comprehensive bash script that automates the entire pipeline from VLR scraping to event extraction.

### Features

- ✅ Scrapes VLR.gg for match metadata
- ✅ Automatically downloads YouTube VODs
- ✅ Processes each map with orchestration
- ✅ Organizes output in clean folder structure
- ✅ Removes VODs after processing to save disk space
- ✅ Colored output with progress tracking

### Prerequisites

1. **valoscribe CLI** must be installed and available in PATH
2. **jq** - JSON processor for parsing metadata
   ```bash
   # macOS
   brew install jq

   # Ubuntu/Debian
   sudo apt-get install jq
   ```
3. **tree** (optional) - for pretty folder structure display
   ```bash
   # macOS
   brew install tree

   # Ubuntu/Debian
   sudo apt-get install tree
   ```

### Usage

```bash
./scripts/process_vlr_series.sh <vlr_url> [output_base_dir]
```

**Arguments:**
- `vlr_url` (required): VLR.gg match URL
- `output_base_dir` (optional): Base directory for output (default: `./series_output`)

### Example

```bash
# Process a match series
./scripts/process_vlr_series.sh "https://www.vlr.gg/542272/nrg-vs-drx-champions-2025" ./my_output

# Using default output directory
./scripts/process_vlr_series.sh "https://www.vlr.gg/542272/nrg-vs-drx-champions-2025"
```

### Output Structure

The script creates an organized folder structure:

```
series_output/
└── nrg_vs_drx/
    ├── series_metadata.json          # Full series metadata
    ├── metadata/                      # Individual map metadata files
    │   ├── map1.json
    │   ├── map2.json
    │   └── map3.json
    ├── map1_haven/                    # Map 1 outputs
    │   ├── metadata.json              # Map-specific metadata
    │   └── output/
    │       ├── frame_states.csv       # Frame-by-frame game state
    │       └── event_log.jsonl        # Extracted events
    ├── map2_bind/                     # Map 2 outputs
    │   ├── metadata.json
    │   └── output/
    │       ├── frame_states.csv
    │       └── event_log.jsonl
    └── map3_ascent/                   # Map 3 outputs
        ├── metadata.json
        └── output/
            ├── frame_states.csv
            └── event_log.jsonl
```

### Pipeline Steps

The script performs these steps automatically:

1. **Scrape VLR Metadata**
   - Fetches match data from VLR.gg
   - Extracts team names, player info, agents, and VOD links

2. **Create Folder Structure**
   - Creates series folder named `team1_vs_team2`
   - Organizes maps into subfolders like `map1_haven`

3. **Split Metadata**
   - Splits series metadata into individual map files
   - Converts to GameStateManager-compatible format

4. **Process Each Map**
   - Downloads YouTube VOD (1080p @ 60fps)
   - Runs orchestration to extract events
   - Saves outputs to map folder
   - Removes VOD to save disk space

5. **Cleanup**
   - Removes temporary files
   - Displays summary and folder structure

### Advanced Options

You can modify the script to customize behavior:

**Download quality:**
```bash
# Line 279: Change --height and --fps
valoscribe download "$VOD_URL" -o "$VOD_DOWNLOAD_DIR" --height 720 --fps 30
```

**Processing FPS:**
```bash
# Line 295: Change --fps
valoscribe orchestrate process-vod "$VOD_FILE" "$MAP_METADATA" \
    --output "$OUTPUT_DIR" \
    --fps 2 \  # Lower FPS for faster processing
    --quiet
```

**Enable debug output:**
```bash
# Line 295: Remove --quiet and add --debug
valoscribe orchestrate process-vod "$VOD_FILE" "$MAP_METADATA" \
    --output "$OUTPUT_DIR" \
    --fps 4 \
    --debug
```

**Enable visual display:**
```bash
# Line 295: Add --show flag
valoscribe orchestrate process-vod "$VOD_FILE" "$MAP_METADATA" \
    --output "$OUTPUT_DIR" \
    --fps 4 \
    --show \  # Display frames during processing
    --quiet
```

### Troubleshooting

**Script fails with "command not found":**
- Ensure `valoscribe` is installed and in PATH
- Install `jq`: `brew install jq` (macOS) or `sudo apt-get install jq` (Linux)

**VOD download fails:**
- Check internet connection
- Verify YouTube URL is accessible
- Try downloading manually with: `valoscribe download <url>`

**Orchestration takes too long:**
- Reduce processing FPS: change `--fps 4` to `--fps 2`
- Process only specific time ranges using `--start` and `--end` flags

**Disk space running low:**
- The script automatically removes VODs after processing
- Check that temp directory cleanup is working
- Manually remove `$OUTPUT_BASE_DIR/.temp` if needed

### Performance Notes

- **Processing time**: ~20-40 minutes per map (depending on VOD length)
- **Disk space**: ~500MB-2GB per map during processing (removed after)
- **Output size**: ~10-50MB per map (CSV + JSONL)

### Example Output

When the script completes, you'll see:

```
========================================================================
[SUCCESS] Pipeline completed successfully!
========================================================================

[INFO] Series: NRG vs DRX
[INFO] Maps processed: 3
[INFO] Output directory: ./series_output/nrg_vs_drx

[INFO] Folder structure:
series_output/nrg_vs_drx
├── series_metadata.json
├── metadata
│   ├── map1.json
│   ├── map2.json
│   └── map3.json
├── map1_haven
│   ├── metadata.json
│   └── output
├── map2_bind
│   ├── metadata.json
│   └── output
└── map3_ascent
    ├── metadata.json
    └── output

[INFO] Next steps:
  - Review event logs: ./series_output/nrg_vs_drx/map*/output/event_log.jsonl
  - Analyze frame states: ./series_output/nrg_vs_drx/map*/output/frame_states.csv
  - Series metadata: ./series_output/nrg_vs_drx/series_metadata.json
```

## Future Enhancements

Potential improvements for the script:

- [ ] Parallel map processing (process multiple maps simultaneously)
- [ ] Resume capability (skip already processed maps)
- [ ] Error recovery (retry failed downloads/processing)
- [ ] Progress bar for long-running operations
- [ ] Email/Slack notifications on completion
- [ ] Automatic upload to cloud storage
- [ ] Statistics summary (total kills, rounds, etc.)

---

## process_vlr_series_dry_run.sh

Test the pipeline without actually downloading VODs or processing them.

### Purpose

- Verify VLR scraping works correctly
- Check that metadata is extracted properly
- Preview the folder structure that will be created
- Estimate total processing time
- Test the pipeline before committing to full processing

### Usage

```bash
./scripts/process_vlr_series_dry_run.sh <vlr_url> [output_base_dir]
```

### What It Does

1. ✅ Scrapes VLR metadata (actually performed)
2. ✅ Splits metadata into map files (actually performed)
3. ✅ Creates folder structure (actually performed)
4. ❌ Downloads VODs (skipped - shows what would be downloaded)
5. ❌ Processes VODs (skipped - shows command that would run)
6. ❌ Removes VODs (skipped)

### Example Output

```
========================================================================
[DRY RUN] Would execute: valoscribe download "https://youtube.com/..." -o ./temp/videos --height 1080 --fps 60
[INFO]   → Would download VOD to temporary directory

[DRY RUN] Would execute: valoscribe orchestrate process-vod <vod_file> "./metadata/map1.json" --output "./output" --fps 4 --quiet
[INFO]   → Would process VOD and extract events
[INFO]   → Output: frame_states.csv, event_log.jsonl

[DRY RUN] Would execute: rm -f <vod_file>
[INFO]   → Would remove VOD to save disk space

[INFO] Estimated processing time: ~35 minutes
```

### When to Use

- **First time using the pipeline**: Test with a dry run to ensure everything works
- **Before long processing jobs**: Verify the VLR URL is valid and metadata scrapes correctly
- **Debugging**: Check folder structure and file organization
- **Planning**: Estimate total processing time before committing

---

## process_single_map.sh

Process a single map when you already have the VOD and metadata files.

### Purpose

- Re-process a map with different settings
- Process maps you've already downloaded separately
- Resume processing after a failure
- Process individual maps from a series

### Usage

```bash
./scripts/process_single_map.sh <vod_file> <metadata_file> <output_dir> [--keep-vod]
```

**Arguments:**
- `vod_file` (required): Path to downloaded VOD file (.mp4)
- `metadata_file` (required): Path to map metadata JSON
- `output_dir` (required): Directory for output files
- `--keep-vod` (optional): Keep VOD after processing (default: delete)

### Examples

```bash
# Process map and delete VOD after
./scripts/process_single_map.sh haven.mp4 map1_metadata.json ./output/map1

# Process map and keep VOD
./scripts/process_single_map.sh haven.mp4 map1_metadata.json ./output/map1 --keep-vod

# Re-process with different settings (edit script first)
./scripts/process_single_map.sh haven.mp4 map1_metadata.json ./output/map1_reprocessed
```

### Output

Creates the following files in the output directory:

```
output/map1/
├── metadata.json              # Copy of input metadata
├── frame_states.csv           # Frame-by-frame game state
└── event_log.jsonl           # Extracted events
```

### When to Use

- **Selective processing**: Only process specific maps from a series
- **Re-processing**: Run orchestration again with different settings
- **Resume after failure**: Continue processing if the main script failed
- **Custom workflows**: Process VODs from sources other than VLR
- **Debugging**: Test orchestration on a single map

### Event Summary

The script automatically prints a summary after processing:

```
[INFO] Event Summary:
  Total events: 1,234
  Rounds: 24
  Kills: 456
  Abilities used: 123
  Ultimates used: 45
```

---

## Contributing

To improve these scripts:

1. Test with various VLR match URLs
2. Report any edge cases or failures
3. Submit improvements via pull request

### Testing Checklist

- [ ] Test with different VLR match URLs (Champions, VCT, etc.)
- [ ] Test with maps that have/don't have VOD links
- [ ] Test dry run mode
- [ ] Test single map processing
- [ ] Test error recovery (interrupted processing)
- [ ] Test with different video qualities
- [ ] Verify folder structure is correct
- [ ] Check event extraction accuracy
