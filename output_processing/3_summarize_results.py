import pandas as pd
from pathlib import Path

# Folder to scan
base_folder = Path("/data-kp/FIP/Analysis/2023/WW034/volume_prediction/2023")

# Collect results
summary = []

# Recursively find all *_filtered_results.csv files
for file in base_folder.rglob("*_filtered_results.csv"):
    filename = file.name
    try:
        # Extract values from filename
        parts = filename.split("_")
        uid_nr = parts[0]  # e.g., FPWW0340432
        timestamp_raw = parts[2]  # e.g., 20230704

        plot_uid = uid_nr[:8]
        print(plot_uid)
        plot_nr = uid_nr[8:]
        print(plot_nr)
        timestamp = f"{timestamp_raw[:4]}-{timestamp_raw[4:6]}-{timestamp_raw[6:]}"  # format YYYY-MM-DD
        print(timestamp)

        # Read CSV
        df = pd.read_csv(file)

        # Compute values
        spike_count = len(df)
        volume_mean = df["volume"].mean()
        volume_sum = df["volume"].sum()

        # Add to summary
        summary.append({
            "file": str(file),
            "plot.UID": plot_uid,
            "plot_nr": plot_nr,
            "timestamp": timestamp,
            "spike_count": spike_count,
            "volume_mean": volume_mean,
            "volume_sum": volume_sum,
        })
    except Exception as e:
        print(f" Error processing {file.name}: {e}")

# Save summary as CSV
summary_df = pd.DataFrame(summary)
summary_df.to_csv("summary_output_2023.csv", index=False)
print(" Summary saved to summary_output.csv")
