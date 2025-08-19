import pandas as pd

df_all = pd.read_csv("/home/zumstego/volume_prediction_fip/output_processing/outliers/all_2023.csv")
print(df_all["overexposed"].dtype)

#save all outliers
outliers = df_all[df_all["overexposed"] == True]
outliers.to_csv("/home/zumstego/volume_prediction_fip/output_processing/outliers/outliers_2023.csv", index=False)

print(outliers["lot_folder"].value_counts())
#discuss which ones we should remove, there are a lot!!

##############################################################
#remove single rows, boarders, etc. 

results = pd.read_csv("/home/zumstego/volume_prediction_fip/output_processing/summary_output_2023.csv")


#2023 lot 1: we need plot 73 - 414
Lot1_2023 = list(range(73,415))
#2023 lot 6: we need plot 451 - 792
Lot6_2023 = list(range(451,793))


selected_plots = results[results["plot_nr"].isin(Lot1_2023) | results["plot_nr"].isin(Lot6_2023)].copy()
selected_plots.to_csv("/home/zumstego/volume_prediction_fip/output_processing/outliers/selected_plots_2023.csv", index=False)


#2024 lot 4: we need plot 469 - 810
#2024 lot 3: we need plot 19 - 360


#remove everything before June or so where some genotypes dont have spikes?



#remove outliers? 
pattern1 = r"/(FPWW\d+_FIP2_\d{8}_\d{6})/"
pattern2 = r"(FPWW\d+_FIP2_\d{8}_\d{6})/"


selected_plots["unique"] = selected_plots["file"].str.extract(pattern1, expand=False)
outliers["unique"] = outliers["rel_path"].str.extract(pattern2, expand=False)
selected_plots_without_outliers = selected_plots[~selected_plots["unique"].isin(outliers["unique"])].copy()

selected_plots_without_outliers.to_csv("/home/zumstego/volume_prediction_fip/output_processing/outliers/selected_plots_2023_without_outliers.csv", index=False)

print(selected_plots_without_outliers["timestamp"].value_counts())
print(selected_plots["timestamp"].value_counts())

