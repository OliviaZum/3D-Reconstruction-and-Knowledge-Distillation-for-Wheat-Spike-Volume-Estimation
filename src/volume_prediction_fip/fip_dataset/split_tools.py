"""
Contains tools to create a random split of data from a vol_mapping file,
or to take over an existing split from an existing dataset into a new dataset.
This needs to be run after exporting something using FIPDataset in order to get the split files.
See main for usage.
"""


from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from volume_prediction_fip.fip_dataset import split_dataset

class VolumeMapping:
    def __init__(self, mapping_path: Path | str):
        mapping_path = Path(mapping_path)
        self.save_mapping_plants = mapping_path / "mapping_all_plants.json"
        self.save_mapping_train = mapping_path / "mapping_train.json"
        self.save_mapping_val = mapping_path / "mapping_val.json"
        self.save_mapping_test = mapping_path / "mapping_test.json"
        self.save_mapping_all_val = mapping_path / "mapping_all_val.json"

    def create_split(self, mapping_file: str | Path, random_split_generator : np.random.Generator, min_sequence_length_train: int,
                     min_sequence_length_test, rel_test_set_size=0.2, rel_val_set_size=0.1):
        df_mapping = pd.read_csv(mapping_file)
        split_dataset.compute_and_store_split(
            df_mapping=df_mapping,
            mapping_plants_path=self.save_mapping_plants,
            mapping_train_path=self.save_mapping_train,
            mapping_val_path=self.save_mapping_val,
            mapping_test_path=self.save_mapping_test,
            mapping_plants_val_path=self.save_mapping_all_val,
            value_column="volume",
            random_generator=random_split_generator,
            rel_test_set_size=rel_test_set_size,
            rel_val_set_size=rel_val_set_size,
            min_view_train=min_sequence_length_train,
            min_view_test=min_sequence_length_test,
            verbose=True,
        )

    def adapt_new_dataset(self, mapping_path_new: Path | str, mapping_file_new: Path | str, min_sequence_length_train: int,
                     min_sequence_length_test, remove_artifical = True, extend_train = True):
        mapping_path_new = Path(mapping_path_new)
        mapping_path_new.mkdir(parents=True, exist_ok=True)

        new_mapping = VolumeMapping(mapping_path_new)

        def create_sub_df_from_indices(new_df: pd.DataFrame, reference_df: str, out_df, name):
            reference_df = pd.read_json(reference_df, orient='index', convert_axes=False, dtype={"plant_id": str})
            sub_df = new_df.loc[new_df.index.intersection(reference_df.index)]
            if remove_artifical:
                sub_df = sub_df.apply(split_dataset.remove_artificial, axis=1)
            sub_df = sub_df[sub_df["images"].apply(lambda x: len(x) >= min_sequence_length_test)]
            missing_count = len(reference_df.index.difference(sub_df.index))
            print(f"# Removed plants {name}: {missing_count}")

            sub_df.to_json(out_df, orient="index")

        def create_exclusive_sub_df(new_df: pd.DataFrame, sequence_of_dfs: List[str], out_df, name):
            all_indices = pd.Index([])
            for df in sequence_of_dfs:
                df = pd.read_json(df, orient='index', convert_axes=False, dtype={"plant_id": str})
                all_indices = all_indices.union(df.index)

            exclusive_df = new_df.loc[~new_df.index.isin(all_indices)]
            exclusive_df = exclusive_df[exclusive_df["images"].apply(lambda x: len(x) >= min_sequence_length_train)]
            print(f"# Plants added {name} (overall): {len(exclusive_df)}")

            exclusive_df.to_json(out_df, orient="index")

        csv_data = pd.read_csv(mapping_file_new)
        df_new = split_dataset.compute_plant_mapping_(csv_data)

        df_new.to_json(new_mapping.save_mapping_plants, orient="index")
        print("The #removed plants indicate how many plants have been removed with respect to the old split, because they do not exist here.")
        create_sub_df_from_indices(df_new, self.save_mapping_test, new_mapping.save_mapping_test, "test")
        create_sub_df_from_indices(df_new, self.save_mapping_val, new_mapping.save_mapping_val, "val")
        if extend_train:
            create_exclusive_sub_df(df_new, [new_mapping.save_mapping_test, new_mapping.save_mapping_val], new_mapping.save_mapping_train, "train")
        else:
            create_sub_df_from_indices(df_new, self.save_mapping_train, new_mapping.save_mapping_train, "train")
        all_val = split_dataset.create_all_val([new_mapping.save_mapping_train, new_mapping.save_mapping_test, new_mapping.save_mapping_val], min_sequence_length_test)
        all_val.to_json(new_mapping.save_mapping_all_val, orient="index")

if __name__ == "__main__":
    # Defines a split. If this split does not yet exist and create_split is called, defines where to store 
    # the split. If it exists, the split defined here can be adapted for a new dataset using adapt_new_dataset.
    #v = VolumeMapping(r"F:\Boxes-ds\segmented_distance_depth\split_without2024")
    v = VolumeMapping(r"unlabeled-5000-depth")


    # Creates a new split based on the vol_mapping file of the dataset
    #v.create_split(r"F:\Boxes-ds\artifical\bestpose_noshift_12\vol_mapping.csv", np.random.default_rng(14), 6, 6)
    v.create_split(r"unlabeled-5000-depth/vol_mapping.csv", np.random.default_rng(14), min_sequence_length_train=6, min_sequence_length_test=6, rel_test_set_size=0.2, rel_val_set_size=0)


    # Adapts an existing split to a new dataset. That is new images/plants are taken to train from the new dataset (if extend_train is true),
    # but test and val are kept with only the plants that where there in the old dataset. 
    #v.adapt_new_dataset(r"F:\Boxes-ds\artifical\bestpose_noshift_12\split_without2024_noextend_6img",
                        #r"F:\Boxes-ds\artifical\bestpose_noshift_12\vol_mapping_6_img.csv",
                        #6, 6, remove_artifical=False, extend_train=True)