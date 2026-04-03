import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read in spike traits and the test predicted and true volumes
traits_df = pd.read_excel("local_stuff_experiments/scans_volumes_traits.xlsx")
volumes_df = pd.read_csv("local_stuff_experiments/test_predictions_self-distill-regulated_transformer_new_mae_rate.pth.csv")
#volumes_df = pd.read_csv("local_stuff_experiments/test_predictions_regulatedtransformer-direct.pth.csv")
#volumes_df = pd.read_csv("local_stuff_experiments/test_predictions_kd_regulated_transformer_new2.pth.csv")
#volumes_df = pd.read_csv("local_stuff_experiments/test_predictions_ensemble.csv")


#merge the two files
traits_volumes_df = volumes_df[["plant_id", "volume_pred_mm3", "volume_real_mm3"]].merge(traits_df[["file_id", "length", "width", "curvature"]], 
                                    left_on="plant_id", 
                                    right_on="file_id", 
                                    how="left")

#add MAE for each spike
traits_volumes_df["MAE"] =(traits_volumes_df["volume_pred_mm3"] - traits_volumes_df["volume_real_mm3"]).abs()


def add_line_and_corr(ax, x, y):
    # remove NaNs
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]
    y = y[mask]

    # regression line
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, linewidth=2)

    # correlation
    r = np.corrcoef(x, y)[0, 1]
    ax.text(
        0.05, 0.95,
        f"r = {r:.3f}",
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=18,
        bbox=dict(boxstyle="round", facecolor="white", alpha=1)
    )


#plot corrlations between MAE and traits
fig, axes = plt.subplots(1,4, figsize=(28,6))
scatter_kwargs = dict(s=10, c="black")

label_size = 24
title_size = 22

#  MAE vs True Volume
axes[0].scatter(traits_volumes_df["volume_real_mm3"], traits_volumes_df["MAE"], **scatter_kwargs)
add_line_and_corr(
    axes[0],
    traits_volumes_df["volume_real_mm3"],
    traits_volumes_df["MAE"])
axes[0].set_xlabel("True Volume (mm³)", fontsize=label_size)
axes[0].set_ylabel("MAE [mm³]", fontsize=label_size)
axes[0].set_ylim(-200, 3200)
#axes[0, 0].set_title("MAE vs True Volume")

#  MAE vs Length
axes[1].scatter(traits_volumes_df["length"], traits_volumes_df["MAE"], **scatter_kwargs)
add_line_and_corr(
    axes[1], 
    traits_volumes_df["length"], 
    traits_volumes_df["MAE"])
axes[1].set_xlabel("Length", fontsize=label_size)
axes[1].set_ylabel("MAE [mm³]", fontsize=label_size)
axes[1].set_ylim(-200, 3200)
#axes[0, 1].set_title("MAE vs Length")

#  MAE vs Width
axes[2].scatter(traits_volumes_df["width"], traits_volumes_df["MAE"], **scatter_kwargs)
add_line_and_corr(axes[2], 
        traits_volumes_df["width"], 
        traits_volumes_df["MAE"])
axes[2].set_xlabel("Width", fontsize=label_size)
axes[2].set_ylabel("MAE [mm³]", fontsize=label_size)
axes[2].set_ylim(-200, 3200)#axes[1, 0].set_title("MAE vs Width")

# MAE vs Curvature
axes[3].scatter(traits_volumes_df["curvature"], traits_volumes_df["MAE"], **scatter_kwargs)
add_line_and_corr(axes[3], 
        traits_volumes_df["curvature"], 
        traits_volumes_df["MAE"])
axes[3].set_xlabel("Curvature", fontsize=label_size)
axes[3].set_ylabel("MAE [mm³]", fontsize=label_size)
axes[3].set_ylim(-200, 3200)#axes[1, 1].set_title("MAE vs Curvature")

for ax in axes.flat:
    ax.tick_params(axis='both', labelsize=20)

plt.tight_layout()

# --- Save figure ---
#plt.savefig("local_stuff_experiments/mae_analysis_distilled_regulated_Transformer.png", dpi=300)
plt.savefig("local_stuff_experiments/all_label_KD.png", dpi=300)

plt.close()


#separate: 

#plot corrlations between MAE and traits
fig, axes = plt.subplots(figsize=(6,5))
scatter_kwargs = dict(s=10, c="black")

label_size = 20

#  MAE vs True Volume
axes.scatter(traits_volumes_df["volume_real_mm3"], traits_volumes_df["MAE"], **scatter_kwargs)
add_line_and_corr(
    axes,
    traits_volumes_df["volume_real_mm3"],
    traits_volumes_df["MAE"])

axes.set_xlabel("True Volume (mm³)", fontsize=label_size)
axes.set_ylabel("MAE [mm³]", fontsize=label_size)
axes.set_ylim(-200, 3200)#axes[0, 0].set_title("MAE vs True Volume")


axes.tick_params(axis='both', labelsize=15)

plt.tight_layout()#

# --- Save figure ---
#plt.savefig("local_stuff_experiments/mae_analysis_distilled_regulated_Transformer.png", dpi=300)
plt.savefig("local_stuff_experiments/mae_label_KD.png", dpi=600)
plt.close()