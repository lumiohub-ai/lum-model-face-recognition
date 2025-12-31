#!/bin/bash
# Filter Log Analysis Script
# Analyzes filter_decisions.jsonl to provide insights on filter performance

LOG_FILE="${1:-/app/logs/filter_decisions.jsonl}"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found: $LOG_FILE"
    echo "Usage: $0 [path/to/filter_decisions.jsonl]"
    exit 1
fi

echo "=========================================="
echo "Filter Performance Analysis"
echo "=========================================="
echo "Log file: $LOG_FILE"
echo "Total entries: $(wc -l < "$LOG_FILE")"
echo ""

# 1. Overall reduction rate
echo "1. OVERALL REDUCTION RATE"
echo "----------------------------"
jq -s 'group_by(.decision) | map({decision: .[0].decision, count: length, percentage: (length / (map(length) | add) * 100)}) | .[]' "$LOG_FILE"
echo ""

# 2. Top 10 rejection reasons
echo "2. TOP 10 REJECTION REASONS"
echo "----------------------------"
jq -s 'map(select(.decision=="FILTER")) | group_by(.reason) | map({reason: .[0].reason, count: length}) | sort_by(-.count) | .[0:10] | .[] | "\(.count)\t\(.reason)"' "$LOG_FILE" -r
echo ""

# 3. Filter stage breakdown
echo "3. FILTER STAGE BREAKDOWN"
echo "----------------------------"
echo "Track lifetime filters:"
jq -s 'map(select(.reason | contains("short_track"))) | length' "$LOG_FILE"

echo "Uncertainty zone filters:"
jq -s 'map(select(.reason | contains("uncertain"))) | length' "$LOG_FILE"

echo "Quality filters (blur/profile/lighting):"
jq -s 'map(select(.reason | contains("blurry") or contains("profile") or contains("lighting") or contains("small") or contains("occluded") or contains("low_quality"))) | length' "$LOG_FILE"

echo "Similarity filters:"
jq -s 'map(select(.reason | contains("similarity_too_high"))) | length' "$LOG_FILE"

echo "Deduplication filters:"
jq -s 'map(select(.reason | contains("duplicate"))) | length' "$LOG_FILE"

echo "Rate limiting filters:"
jq -s 'map(select(.reason | contains("rate_limited"))) | length' "$LOG_FILE"
echo ""

# 4. Quality score distribution
echo "4. QUALITY SCORE DISTRIBUTION"
echo "----------------------------"
echo "Sent to dashboard:"
jq -s 'map(select(.decision=="SEND")) | {avg: (map(.quality_score) | add / length), min: (map(.quality_score) | min), max: (map(.quality_score) | max)}' "$LOG_FILE"

echo "Filtered out:"
jq -s 'map(select(.decision=="FILTER")) | {avg: (map(.quality_score) | add / length), min: (map(.quality_score) | min), max: (map(.quality_score) | max)}' "$LOG_FILE"
echo ""

# 5. Similarity distribution
echo "5. SIMILARITY DISTRIBUTION"
echo "----------------------------"
echo "All tracks:"
jq -s '{avg: (map(.similarity) | add / length), min: (map(.similarity) | min), max: (map(.similarity) | max)}' "$LOG_FILE"

echo "Sent to dashboard:"
jq -s 'map(select(.decision=="SEND")) | {avg: (map(.similarity) | add / length), min: (map(.similarity) | min), max: (map(.similarity) | max)}' "$LOG_FILE"
echo ""

# 6. Per-camera breakdown
echo "6. PER-CAMERA BREAKDOWN"
echo "----------------------------"
jq -s 'group_by(.camera) | map({camera: .[0].camera, total: length, sent: (map(select(.decision=="SEND")) | length), filtered: (map(select(.decision=="FILTER")) | length)}) | map({camera: .camera, total: .total, sent: .sent, filtered: .filtered, reduction_pct: ((.filtered / .total) * 100)}) | .[] | "\(.camera):\t\(.total) total, \(.sent) sent, \(.filtered) filtered (\(.reduction_pct | floor)% reduction)"' "$LOG_FILE" -r
echo ""

# 7. Track lifetime statistics
echo "7. TRACK LIFETIME STATISTICS"
echo "----------------------------"
echo "Sent to dashboard:"
jq -s 'map(select(.decision=="SEND")) | {avg_lifetime: (map(.track_lifetime) | add / length), min: (map(.track_lifetime) | min), max: (map(.track_lifetime) | max)}' "$LOG_FILE"

echo "Filtered out:"
jq -s 'map(select(.decision=="FILTER")) | {avg_lifetime: (map(.track_lifetime) | add / length), min: (map(.track_lifetime) | min), max: (map(.track_lifetime) | max)}' "$LOG_FILE"
echo ""

# 8. Deduplication effectiveness
echo "8. DEDUPLICATION EFFECTIVENESS"
echo "----------------------------"
echo "Same-camera duplicates:"
jq -s 'map(select(.reason | contains("duplicate_same_camera"))) | length' "$LOG_FILE"

echo "Cross-camera duplicates:"
jq -s 'map(select(.reason | contains("duplicate_cross_camera"))) | length' "$LOG_FILE"

echo "Quality upgrades (replaced lower quality cache entry):"
jq -s 'map(select(.reason | contains("duplicate_upgraded"))) | length' "$LOG_FILE"
echo ""

# 9. Shadow mode status
echo "9. SHADOW MODE STATUS"
echo "----------------------------"
SHADOW_MODE=$(jq -s '.[0].shadow_mode' "$LOG_FILE")
if [ "$SHADOW_MODE" == "true" ]; then
    echo "⚠️  SHADOW MODE IS ENABLED - Filters are NOT being applied!"
    echo "    All tracks are being sent to dashboard for testing."
    echo "    Set shadow_mode=false to enable actual filtering."
else
    echo "✅ SHADOW MODE IS DISABLED - Filters are actively being applied."
fi
echo ""

# 10. Recommendations
echo "10. RECOMMENDATIONS"
echo "----------------------------"

# Check if any single reason dominates
DOMINANT_REASON=$(jq -s 'map(select(.decision=="FILTER")) | group_by(.reason) | map({reason: .[0].reason, count: length, pct: (length / (map(length) | add) * 100)}) | sort_by(-.pct) | .[0]' "$LOG_FILE")
DOMINANT_PCT=$(echo "$DOMINANT_REASON" | jq '.pct')

if (( $(echo "$DOMINANT_PCT > 50" | bc -l) )); then
    REASON=$(echo "$DOMINANT_REASON" | jq -r '.reason')
    echo "⚠️  Warning: One rejection reason dominates (>50%): $REASON"
    echo "    This may indicate a systematic issue rather than a filtering problem."

    if [[ "$REASON" == *"blurry"* ]]; then
        echo "    → Suggestion: Check camera focus or lower MIN_LAPLACIAN_VARIANCE"
    elif [[ "$REASON" == *"profile"* ]]; then
        echo "    → Suggestion: Check camera angle or lower MIN_FRONTALITY_SCORE"
    elif [[ "$REASON" == *"small"* ]]; then
        echo "    → Suggestion: Check camera positioning or lower MINIMUM_FACE_SIZE"
    fi
else
    echo "✅ No dominant single rejection reason - filters are well-balanced."
fi

# Check dedup effectiveness
TOTAL_FILTERED=$(jq -s 'map(select(.decision=="FILTER")) | length' "$LOG_FILE")
DEDUP_COUNT=$(jq -s 'map(select(.reason | contains("duplicate"))) | length' "$LOG_FILE")
DEDUP_RATE=$(echo "scale=2; ($DEDUP_COUNT / $TOTAL_FILTERED) * 100" | bc)

if (( $(echo "$DEDUP_RATE < 10" | bc -l) )); then
    echo "⚠️  Dedup effectiveness is low (<10%)"
    echo "    → Suggestion: Lower DEDUP_SIMILARITY_THRESHOLD to 0.80"
else
    echo "✅ Dedup is working well ($DEDUP_RATE% of filters)"
fi

# Check quality distribution
AVG_SENT_QUALITY=$(jq -s 'map(select(.decision=="SEND")) | map(.quality_score) | add / length' "$LOG_FILE")

if (( $(echo "$AVG_SENT_QUALITY < 0.65" | bc -l) )); then
    echo "⚠️  Average quality of sent faces is low (<0.65)"
    echo "    → Suggestion: Increase MIN_TRACK_QUALITY_SCORE"
else
    echo "✅ Good quality faces being sent (avg: $AVG_SENT_QUALITY)"
fi

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
