import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ----------------------------
# Load JSON
# ----------------------------
with open("/home/zumstego/volume_prediction_fip/splitting_folders/segmented_distance_depth_new2/split_without2024/mapping_all_plants.json", "r") as f:
    data = json.load(f)

rows = []

for sample_key, entry in data.items():
    if not entry.get("labeled", False):
        continue

    volume = entry["volume"]

    pose_file = entry["pose_file"][0]  # e.g. 2023_06_08_13_11_Lot1.json
    parts = pose_file.split("_")

    year = int(parts[0])
    sampling_date = f"{parts[0]}-{parts[1]}-{parts[2]}"

    rows.append({
        "sample_key": sample_key,
        "volume": volume,
        "sampling_date": sampling_date,
        "year": year
    })

df = pd.DataFrame(rows)

# ----------------------------
# Print group info
# ----------------------------
print("Samples per group:")
print(df.groupby(['year', 'sampling_date']).size())

print("\nMean volume per group:")
print(df.groupby(['year', 'sampling_date'])['volume'].mean().round(3))

# ----------------------------
# Merge 2023 + 2024 by stage
# ----------------------------

# Convert to datetime
df["date_obj"] = pd.to_datetime(df["sampling_date"])

# Rank sampling dates within each year → stage 1,2,3
df["stage"] = (
    df.sort_values("date_obj")
      .groupby("year")["date_obj"]
      .rank(method="dense")
      .astype(int)
)

stage_stats = df.groupby("stage")["volume"].agg(["mean", "std", "count"])

print("\nMean and std per stage:")
print(stage_stats.round(3))

# ----------------------------
# Plot merged stages
# ----------------------------

def plot_merged_stages(ax, data):

    stage_labels = {
        1: "Early sampling",
        2: "Mid sampling",
        3: "Late sampling"
    }

    for stage in sorted(data["stage"].unique()):
        subset = data[data["stage"] == stage]
        n = len(subset)
        mean_val = subset["volume"].mean()

        sns.histplot(
            data=subset,
            x="volume",
            ax=ax,
            bins=30,
            stat="density",
            element="step",
            fill=True,
            alpha=0.15,
        )

        line = sns.kdeplot(
            data=subset,
            x="volume",
            ax=ax,
            label=f"{stage_labels[stage]} (n={n})",
            fill=False
        )

        color = line.lines[-1].get_color()

        ax.axvline(
            x=mean_val,
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            color=color
        )

    ax.set_xlim(0, 11000)
    ax.set_xlabel("Measured Volume [mm³]", fontsize=22)
    ax.set_ylabel("Density", fontsize=22)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True)
    ax.legend(fontsize=15)


# ----------------------------
# Create single plot
# ----------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

plot_merged_stages(ax, df)

plt.tight_layout()
plt.savefig(
    'local_stuff_experiments/plot_distribution_by_stage.png',
    dpi=300
)


import json
import pandas as pd
import numpy as np

# load json
with open("/home/zumstego/volume_prediction_fip/splitting_folders/segmented_distance_depth_new2/split_without2024/mapping_test.json", "r") as f:
    data = json.load(f)

# convert to dataframe
rows = []
for spike_id, spike_data in data.items():
    genotype = spike_data["genotype_id"]
    volume = spike_data["volume"]
    
    rows.append({
        "spike_id": spike_id,
        "genotype": genotype,
        "volume": volume
    })

df = pd.DataFrame(rows)

# -------------------------------------------------
# 1) Mean volume over ALL spikes
# -------------------------------------------------
overall_mean_volume = df["volume"].mean()

# -------------------------------------------------
# 2) Standard deviation within each genotype
# -------------------------------------------------
std_within_genotype = df.groupby("genotype")["volume"].std()

# -------------------------------------------------
# 3) Mean of genotype standard deviations
# -------------------------------------------------
mean_std_within_genotype = std_within_genotype.mean()

print("Mean volume over all spikes:")
print(overall_mean_volume)

print("\nMean standard deviation within genotype:")
print(mean_std_within_genotype)

# ------------------------------------------
# Spread across genotypes
# (std of genotype means)
# ------------------------------------------
mean_per_genotype = df.groupby("genotype")["volume"].mean()

std_across_genotypes = mean_per_genotype.std()

print("\nStandard deviation across genotype means:")
print(std_across_genotypes)

overall_mean = df["volume"].mean()
overall_std = df["volume"].std()

print("Overall mean spike volume:", overall_mean)
print("Overall standard deviation:", overall_std)


import pandas as pd

# -------------------------------------------------
# 1) Read file  (change to read_excel if needed)
# -------------------------------------------------
df = pd.read_csv("local_stuff_experiments/test_predictions_self-distill-regulated_transformer_new_mae_rate_original.pth.csv")
# df = pd.read_excel("your_test_predictions.xlsx")


# ---------------------------
# 2) Extract genotype
# ---------------------------
df["genotype"] = df["plant_id"].apply(
    lambda x: "_".join(str(x).split("_")[:2])
)

# ---------------------------
# 3) Compute genotype means
# ---------------------------
genotype_means = df.groupby("genotype").agg(
    true_mean=("volume_real_mm3", "mean"),
    pred_mean=("volume_pred_mm3", "mean")
)

# ---------------------------
# 4) MAE per genotype group
# ---------------------------
genotype_means["genotype_MAE"] = (
    genotype_means["pred_mean"] -
    genotype_means["true_mean"]
).abs()

print(genotype_means)

# ---------------------------
# 5) Overall genotype-level MAE
# ---------------------------
overall_genotype_MAE = genotype_means["genotype_MAE"].mean()

print("\nOverall MAE between genotype means:")
print(overall_genotype_MAE)