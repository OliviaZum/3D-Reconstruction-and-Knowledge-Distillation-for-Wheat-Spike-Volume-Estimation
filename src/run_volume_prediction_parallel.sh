#!/bin/bash
export PYTHONPATH=$(pwd)

#INPUT_PARENT="../../../../data-kp/FIP/Analysis/2024/WW036/debayered"
#OUTPUT_PARENT="../../../../data-kp/FIP/Analysis/2023/WW034/volume_prediction/2024"
INPUT_PARENT="../../../../data-kp/FIP/Analysis/2023/WW034/debayered"
OUTPUT_PARENT="../../../../data-kp/FIP/Analysis/2023/WW034/volume_prediction/2023"
#INPUT_PARENT="../2_Messung_2024"
#OUTPUT_PARENT="../2_Output_2024"
mkdir -p log_files
SUMMARY_FILE="log_files/summary_$(date +%Y%m%d_%H%M%S).csv"



# Header for CSV
echo "plot_name,status,duration_seconds,missing_files" > "$SUMMARY_FILE"

# Function to process a plot folder
process_plot() {
    INPUT_SUBDIR="$1"
    OUTPUT_PARENT="$2"
    SUMMARY_FILE="$3"

    PLOT_NAME=$(basename "$INPUT_SUBDIR")

    # Extract date from folder name (e.g., FPWW0360218_FIP2_20240704_133657)
    DATE_STR=$(echo "$PLOT_NAME" | grep -oE '[0-9]{8}_[0-9]{6}' | cut -d_ -f1)

    # Skip if date is before 2024-05-15
    if [[ "$DATE_STR" < "20240515" ]]; then
        #echo "$PLOT_NAME,skipped,0,older_than_threshold" >> "$SUMMARY_FILE"
        echo "Skipping $PLOT_NAME (date $DATE_STR is before 2024-05-15)"
        return
    fi

    PARENT_NAME=$(basename $(dirname "$INPUT_SUBDIR"))
    OUTPUT_SUBDIR="$OUTPUT_PARENT/$PARENT_NAME/$PLOT_NAME"
    mkdir -p "$OUTPUT_SUBDIR"
    POSES_FILE="$OUTPUT_SUBDIR/poses_scaled.json"
    LOG_FILE="$OUTPUT_SUBDIR/log.txt"

    PREDICTED_CSV="$OUTPUT_SUBDIR/${PLOT_NAME}_predicted_volume.csv"
    FILTERED_CSV="$OUTPUT_SUBDIR/${PLOT_NAME}_filtered_results.csv"


    START=$(date +%s.%N)

    if [[ -f "$PREDICTED_CSV" && -f "$FILTERED_CSV" ]]; then
        echo "$PLOT_NAME,skipped,0," >> "$SUMMARY_FILE"
        echo "Skipping $PLOT_NAME (already processed)"
        return
    fi

    if [[ ! -f "$POSES_FILE" ]]; then
        echo "$PLOT_NAME,error,0,poses_scaled.json" >> "$SUMMARY_FILE"
        echo "Warning: Missing poses for $PLOT_NAME"
        return
    fi

    mkdir -p "$OUTPUT_SUBDIR"
    echo "Processing $PLOT_NAME..."

    python -m volume_prediction_fip.volume_prediction \
        -c "$POSES_FILE" \
        -i "$INPUT_SUBDIR" \
        -od "$OUTPUT_SUBDIR" \
        -mv 10 \
        > "$LOG_FILE" 2>&1

    END=$(date +%s.%N)
    DURATION=$(echo "$END - $START" | bc)

    MISSING=()
    [[ ! -f "$PREDICTED_CSV" ]] && MISSING+=("predicted_volume.csv")
    [[ ! -f "$FILTERED_CSV" ]] && MISSING+=("filtered_results.csv")

    if [[ ${#MISSING[@]} -eq 0 ]]; then
        echo "$PLOT_NAME,ok,$DURATION," >> "$SUMMARY_FILE"
        echo "Done: $PLOT_NAME"
    else
        echo "$PLOT_NAME,incomplete,$DURATION,${MISSING[*]}" >> "$SUMMARY_FILE"
        echo "Incomplete: $PLOT_NAME (missing ${MISSING[*]})"
    fi
}

export -f process_plot

# Feed input folders to parallel
find "$INPUT_PARENT" -mindepth 2 -maxdepth 2 -type d | \
    parallel -j 4 process_plot {} "$OUTPUT_PARENT" "$SUMMARY_FILE"
