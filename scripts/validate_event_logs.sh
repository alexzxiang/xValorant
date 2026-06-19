#!/bin/bash

echo "Event Log Validation"
echo "===================="
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERIES_OUTPUT_DIR="$PROJECT_ROOT/series_output"

total_logs=0
round_mismatch_count=0
missing_match_start_count=0
missing_match_end_count=0
score_too_low_count=0
score_diff_too_small_count=0

declare -a all_issues

while IFS= read -r event_log; do
    total_logs=$((total_logs + 1))

    # Extract match/map info
    match_map=$(echo "$event_log" | sed "s|.*$SERIES_OUTPUT_DIR/||" | sed 's|/output/event_log.jsonl||')

    issues=()

    # Check 1: Round start/end counts
    round_starts=$(grep -c '"type": "round_start"' "$event_log" 2>/dev/null)
    round_ends=$(grep -c '"type": "round_end"' "$event_log" 2>/dev/null)

    if [ "$round_starts" -ne "$round_ends" ]; then
        round_mismatch_count=$((round_mismatch_count + 1))
        issues+=("round_start/end mismatch ($round_starts vs $round_ends)")
    fi

    # Check 2: Match start presence
    match_starts=$(grep -c '"type": "match_start"' "$event_log" 2>/dev/null)

    if [ "$match_starts" -eq 0 ]; then
        missing_match_start_count=$((missing_match_start_count + 1))
        issues+=("missing match_start")
    fi

    # Check 3: Match end presence
    match_ends=$(grep -c '"type": "match_end"' "$event_log" 2>/dev/null)

    if [ "$match_ends" -eq 0 ]; then
        missing_match_end_count=$((missing_match_end_count + 1))
        issues+=("missing match_end")
    fi

    # Check 4: Final score validation
    if [ "$round_ends" -gt 0 ]; then
        last_round=$(grep '"type": "round_end"' "$event_log" | tail -1)

        if [ -n "$last_round" ]; then
            score_team1=$(echo "$last_round" | jq -r '.score_team1' 2>/dev/null)
            score_team2=$(echo "$last_round" | jq -r '.score_team2' 2>/dev/null)

            if [[ "$score_team1" =~ ^[0-9]+$ ]] && [[ "$score_team2" =~ ^[0-9]+$ ]]; then
                max_score=$score_team1
                if [ "$score_team2" -gt "$max_score" ]; then
                    max_score=$score_team2
                fi

                score_diff=$((score_team1 - score_team2))
                if [ "$score_diff" -lt 0 ]; then
                    score_diff=$((-score_diff))
                fi

                # Check if winner has at least 13 rounds
                if [ "$max_score" -lt 13 ]; then
                    score_too_low_count=$((score_too_low_count + 1))
                    issues+=("final score < 13 ($score_team1-$score_team2)")
                fi

                # Check if winner won by at least 2
                if [ "$score_diff" -lt 2 ]; then
                    score_diff_too_small_count=$((score_diff_too_small_count + 1))
                    issues+=("score diff < 2 ($score_team1-$score_team2)")
                fi
            fi
        fi
    fi

    if [ ${#issues[@]} -gt 0 ]; then
        issue_str=$(IFS="; "; echo "${issues[*]}")
        all_issues+=("$match_map: $issue_str")
    fi

done < <(find "$SERIES_OUTPUT_DIR" -name "event_log.jsonl" -type f | sort)

passing=$((total_logs - ${#all_issues[@]}))

echo "RESULTS"
echo "======="
echo "Total event logs checked: $total_logs"
echo "Logs passing all checks: $passing"
echo ""
echo "ISSUE BREAKDOWN:"
echo "  - Round start/end mismatches: $round_mismatch_count"
echo "  - Missing match_start: $missing_match_start_count"
echo "  - Missing match_end: $missing_match_end_count"
echo "  - Final score < 13: $score_too_low_count"
echo "  - Score difference < 2: $score_diff_too_small_count"
echo ""
echo "Total logs with issues: ${#all_issues[@]}"
echo ""

if [ ${#all_issues[@]} -gt 0 ]; then
    echo "FAILED LOGS:"
    echo "============"
    for log in "${all_issues[@]}"; do
        echo "  - $log"
    done
else
    echo "✓ ALL LOGS PASSED ALL VALIDATION CHECKS"
fi
