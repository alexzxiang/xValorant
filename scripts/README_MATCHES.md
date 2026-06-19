# Managing Match URLs

The parallel processing scripts now read match URLs from `matches.txt` instead of hardcoding them in the script.

## Quick Start

### 1. View Current Matches

```bash
cat scripts/matches.txt
```

Currently includes:
- **34 matches** from Valorant Champions 2025
- **19 matches** from VCT 2025 Americas Ascension
- **12 matches** from VCT 2025 EMEA Ascension

**Total: 65 matches**

### 2. Run Processing

```bash
# Local (7 parallel jobs)
cd scripts
./process_all_series_parallel.sh

# Cloud (48 parallel jobs)
./cloud/run_cloud_processing.sh
```

---

## File Format

`matches.txt` format:
```
# Comments start with #
# Blank lines are ignored

https://www.vlr.gg/MATCH_ID/team1-vs-team2-event
https://www.vlr.gg/MATCH_ID/team1-vs-team2-event
...
```

---

## Adding New Matches

### Method 1: Edit File Directly

```bash
# Edit matches.txt
nano scripts/matches.txt

# Add new URLs at the end
https://www.vlr.gg/NEW_MATCH_ID/team1-vs-team2-event
```

### Method 2: Append to File

```bash
# Add a single match
echo "https://www.vlr.gg/NEW_MATCH_ID/..." >> scripts/matches.txt

# Add multiple matches
cat >> scripts/matches.txt <<EOF
https://www.vlr.gg/MATCH1/...
https://www.vlr.gg/MATCH2/...
https://www.vlr.gg/MATCH3/...
EOF
```

### Method 3: Extract from Event Page

Get all matches from a VLR event page:

1. Go to the event page (e.g., `https://www.vlr.gg/event/matches/EVENT_ID/?series_id=all`)
2. Right-click on a match → Inspect
3. Find the match links in format `/MATCH_ID/team1-vs-team2`
4. Prepend with `https://www.vlr.gg`

Or use the VLR scraper if you've extended it to extract match lists.

---

## Processing Subsets

### Process Specific Matches

Create a temporary file with only the matches you want:

```bash
# Create subset
head -n 10 scripts/matches.txt > scripts/matches_test.txt

# Edit script to use subset
# In process_all_series_parallel.sh, change:
MATCHES_FILE="$(dirname "${BASH_SOURCE[0]}")/matches_test.txt"

# Run
./scripts/process_all_series_parallel.sh
```

### Skip Already Processed

The script automatically skips matches that already have output:

```bash
# First run (processes all 65)
./scripts/process_all_series_parallel.sh

# Second run (skips completed, processes failed/new only)
./scripts/process_all_series_parallel.sh
```

Check logs to see which were skipped:
```
[Map 1] Output already exists, skipping processing
  - series_output/542272_nrg_vs_fnatic/map1_haven/output/event_log.jsonl
```

---

## Cloud Processing

### Upload to Cloud

```bash
# Upload matches.txt along with scripts
scp scripts/matches.txt ubuntu@instance:/processing/scripts/
scp scripts/cloud/run_cloud_processing.sh ubuntu@instance:/processing/

# On cloud instance
cd /processing
./run_cloud_processing.sh
```

The cloud script automatically finds `matches.txt` in the parent directory.

---

## Current Match List

### Valorant Champions 2025 (34 matches)
- 8 Opening matches
- 4 Winners matches
- 4 Elimination matches
- 4 Decider matches
- 4 Upper Bracket Quarterfinals
- 2 Lower Round 1
- 2 Upper Bracket Semifinals
- 2 Lower Round 2
- 1 Upper Bracket Finals
- 1 Lower Round 3
- 1 Lower Bracket Finals
- 1 Grand Finals

### VCT 2025 Americas Ascension (19 matches)
- 3 Main Event matches
- 4 Round 1
- 4 Round 2
- 2 Round 3
- 6 Playoff matches

### VCT 2025 EMEA Ascension (12 matches)
- 4 Opening matches
- 4 Winners/Elimination matches
- 2 Decider matches
- 2 Semifinals

---

## Estimated Processing Time

**Local (14-core MacBook Pro, 7 parallel jobs):**
- 65 matches × ~2 maps avg × ~15 min = **~28 hours**

**Cloud (c7i.16xlarge, 48 parallel jobs):**
- 65 matches / 48 parallel = ~2 batches
- 2 batches × ~30 min (longest match) = **~1 hour**

**Cost (AWS Spot):**
- c7i.16xlarge spot: $0.50/hr × 1 hr = **$0.50**

---

## Troubleshooting

### "No match URLs found in matches.txt"

**Cause:** File is empty or all lines are comments

**Solution:**
```bash
# Check file content
cat scripts/matches.txt

# Verify it has URLs
grep -v '^#' scripts/matches.txt | grep -v '^$' | wc -l
# Should show: 65
```

### "matches.txt not found"

**Cause:** Running from wrong directory or file moved

**Solution:**
```bash
# Check current directory
pwd
# Should be: .../valoscribev4

# Verify file exists
ls scripts/matches.txt

# Run from correct location
cd /path/to/valoscribev4
./scripts/process_all_series_parallel.sh
```

### Want to process only new matches

**Option 1: Comment out processed matches**

```bash
# Edit matches.txt
nano scripts/matches.txt

# Add # before processed matches
# https://www.vlr.gg/542272/... (already done)
https://www.vlr.gg/NEW_MATCH/... (new)
```

**Option 2: Create new file**

```bash
# Extract only new matches
cat > scripts/matches_new.txt <<EOF
https://www.vlr.gg/NEW1/...
https://www.vlr.gg/NEW2/...
EOF

# Update script to use new file
# (Change MATCHES_FILE variable)
```

---

## Tips

### 1. Keep matches.txt Organized

Use comments to group matches:

```
# ========== Event Name ==========

# Round 1
https://...
https://...

# Round 2
https://...
```

### 2. Track Processing Status

Add comments after processing:

```
https://www.vlr.gg/542272/... # ✓ Done 2025-01-10
https://www.vlr.gg/542279/... # ✗ Failed (no VOD)
https://www.vlr.gg/542278/... # → Processing...
```

### 3. Backup Before Large Changes

```bash
cp scripts/matches.txt scripts/matches_backup.txt
```

### 4. Validate URLs

Check all URLs are valid:

```bash
grep -v '^#' scripts/matches.txt | grep -v '^$' | grep -v 'vlr.gg' && echo "Found invalid URL!"
```

---

## Next Steps

1. **Review matches.txt** - Verify the 65 matches are correct
2. **Test with subset** - Process 2-3 matches first to verify
3. **Run full batch** - Process all 65 matches
4. **Add more events** - Extend matches.txt with other tournaments

For full documentation, see:
- `scripts/cloud/README.md` - Cloud processing guide
- `scripts/cloud/QUICKSTART.md` - Quick cloud setup
