#!/bin/bash

INPUT_PARENT="../FIP_test_config"
OUTPUT_PARENT="../FIP_output_config"

for INPUT_SUBDIR in "$INPUT_PARENT"/*/; do
    PLOT_NAME=$(basename "$INPUT_SUBDIR")
    CONFIG_FILE="$INPUT_SUBDIR/poses_scaled.json"
    OUTPUT_SUBDIR="$OUTPUT_PARENT/$PLOT_NAME"
    LOG_FILE="log_files/$OUTPUT_SUBDIR/log.txt"

    PREDICTED_CSV="$OUTPUT_SUBDIR/predicted_volume.csv"
    FILTERED_CSV="$OUTPUT_SUBDIR/filtered_results.csv"

    # Checkpoint: Skip if both expected output files exist
    if [[ -f "$PREDICTED_CSV" && -f "$FILTERED_CSV" ]]; then
        echo "Skipping $PLOT_NAME (output files already exist)"
        continue
    fi

    echo "Checking poses: $CONFIG_FILE"

    if [ -f "$CONFIG_FILE" ]; then
        mkdir -p "$OUTPUT_SUBDIR"
        echo "Processing $PLOT_NAME..."

        python -m volume_prediction_fip.volume_prediction \
            -c "$CONFIG_FILE" \
            -i "$INPUT_SUBDIR" \
            -od "$OUTPUT_SUBDIR" \
            -mv 10 \
            > "$LOG_FILE" 2>&1

        # Post-run check for output files
        if [[ -f "$PREDICTED_CSV" && -f "$FILTERED_CSV" ]]; then
            echo "Done: $PLOT_NAME"
        else
            echo "Incomplete output for $PLOT_NAME" | tee -a "$LOG_FILE"
            if [ ! -f "$PREDICTED_CSV" ]; then
                echo "Missing: $PREDICTED_CSV" >> "$LOG_FILE"
            fi
            if [ ! -f "$FILTERED_CSV" ]; then
                echo "Missing: $FILTERED_CSV" >> "$LOG_FILE"
            fi
        fi
    else
        echo "Warning: Missing config file for $PLOT_NAME"
    fi
done
