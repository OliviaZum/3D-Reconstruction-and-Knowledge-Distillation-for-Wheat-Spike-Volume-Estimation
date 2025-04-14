"""
A set of functions used to manage datasets.
(Compute a random split; remove images and or plants; ...)
Not really thought for direct usage, it is used by split_tools
"""

import numpy as np
import pandas as pd
import warnings

from pathlib import Path
from typing import List, Tuple, Dict

def get_plant_constant_columns():
    return ["plant_id", "volume", "labeled"]

def remove_images(row: pd.Series, mask):
    for col in row.index.tolist():
        if col not in get_plant_constant_columns():
            row[col] = np.array(row[col])[mask].tolist()
    return row

def remove_artificial(row : pd.Series):
    mask = ~np.array(row['artificial_mask'])
    return remove_images(row, mask)

def remove_real(row: pd.Series):
    mask = np.array(row['artificial_mask'])
    return remove_images(row, mask)

def remove_2024lastday(df_new):
    input("Removing last day of 2024. Press enter to confirm. (Artifical images are kept)")
    # Filter last day of 2024
    def is_2024_last(row):
        range_, row_, p_ = row.name.split("_")
        range_, row_, p_ = int(range_), int(row_), int(p_)
        return (15 <= row_ <= 19 and p_ in [6, 7, 8, 9, 10])

    lastday = df_new.apply(is_2024_last, axis=1)
    df_new[lastday] = df_new[lastday].apply(remove_real, axis=1)
    df_new = df_new[df_new["images"].apply(lambda x: len(x) > 0)]
    return df_new

def compute_plant_mapping_(df_mapping : pd.DataFrame) -> pd.DataFrame:
    """Create a dataframe with plant_id -> list of images,
    plant_id -> list of artificial indicator and plant_id -> value mappings.
    Using the `df_mapping` which is supposed to be a image_name -> volume and
    image_name -> plant_id mapping.
    
    Images in `df_mapping` with the 'img_name' ending on 'b' or 'c',
    are considered artificial images.

    Args:
        df_mapping (DataFrame): mapping file with image to volume mappings
        value_column (str): Name of the calude column.
    
    Raises:
        KeyError : If in `df_mapping` no column is named `value_column` or no column
            in `df_mapping` is named 'img_name'.
    
    Warnings:
        Warns if multiple different volumes are measured for one plant_id. 
        
    Examples:
        >>> df_mapping
                  img_name  volume plant_id  artificial
        0  10_10_1_2_b.jpg    5000  10_10_1        True
        1  10_10_1_3_a.jpg    5000  10_10_1       False
        2   10_9_2_2_c.jpg    4000   10_9_2        True
        3   10_9_2_3_a.jpg    4000   10_9_2       False
        4    8_9_2_2_c.jpg    3000    8_9_2        True
        5    8_9_2_3_a.jpg    3000    8_9_2       False
        >>> compute_plant_mapping_(df_mapping=df_mapping, value_column='volume')
                volume                              images artificial_mask
        10_10_1  5000.0  [10_10_1_2_b.jpg, 10_10_1_3_a.jpg]   [True, False]
        10_9_2   4000.0    [10_9_2_2_c.jpg, 10_9_2_3_a.jpg]   [True, False]
        8_9_2    3000.0      [8_9_2_2_c.jpg, 8_9_2_3_a.jpg]   [True, False]

    """
    
    # compute plant_ids from image names
    plant_id = df_mapping['img_name'].str.extract(r'^(\d{1,2}_\d{1,2}_\d{1,4})_\d{1,3}(_\d{1,3})?_(\w)')
    df_mapping['plant_id'] = plant_id.iloc[:,0]
    artificial_grayscale = plant_id.iloc[:,2].str.match('b')
    artificial_color = plant_id.iloc[:,2].str.match('c')
    df_mapping['artificial'] = artificial_color | artificial_grayscale

    # get uique plant_ids and assign plant_index
    value_column = "volume"
    grouping = df_mapping.loc[:,[value_column, "plant_id"]].groupby(by='plant_id')
    var = grouping.var()
    if var.shape[0] > 1 and not ((var.loc[:,value_column] < 0.000001) | (var.loc[:, value_column].isna())).all():
        warnings.warn(f"Some plants have inconsitent values.")

    df_unique_mapping = grouping.mean()
    df_unique_mapping = df_unique_mapping.rename_axis('plant_id').reset_index() # create plant_id
    df_unique_mapping = df_unique_mapping.rename_axis('plant_index').reset_index() # create plant_index column
    
    # create dictonary with plant -> (value, [images], [artificial_mask]) mapping
    plants : Dict[str, Dict[str, any]] = {}
    df_mapping.rename(columns={"img_name": "images", "artificial": "artificial_mask"}, inplace=True)
    for _ , current_plant in df_unique_mapping.iterrows():
        slic_ = df_mapping.loc[:,'plant_id'].eq(current_plant['plant_id'])
        current_mappings = df_mapping.loc[slic_]
        current_id = current_plant['plant_id']
        temp_dict = {}
        temp_dict[value_column] = current_plant[value_column]
        temp_dict["labeled"] = not pd.isna(current_plant[value_column])
        
        for name in current_mappings.columns:
            if name in get_plant_constant_columns():
                continue
            temp_dict[name] = current_mappings.loc[:,name].to_list()
        plants[current_id] = temp_dict

    df_plants = pd.DataFrame.from_dict(plants, orient='index').sort_index()
    df_plants = remove_2024lastday(df_plants)

    return df_plants

def create_all_val(dfs: List[Path | str | pd.DataFrame], min_images: int) -> pd.DataFrame:
    for i in range(len(dfs)):
        if isinstance(dfs[i], str) or isinstance(dfs[i], Path):
            dfs[i] = pd.read_json(dfs[i], orient='index', convert_axes=False, dtype={"plant_id": str})

    all_val = pd.concat(dfs, axis=0)

    if "artificial_mask" in all_val.columns:
        all_val = all_val.apply(remove_artificial, axis=1)
    all_val = all_val[all_val["images"].apply(lambda x: len(x) >= min_images)]
    assert all_val["labeled"].all(), "There are unlabeled plants in all val (should be marked as having all artifical_images)"

    return all_val

def split_(
        plant_mapping : pd.DataFrame, 
        test_set_size : int,
        val_set_size : int,
        random_generator : np.random.Generator,
        min_view_train: int = 4,
        min_view_test: int = 6,
        verbose : bool = False,
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to split the `plant_mapping` in training, validation and test mapping
    
    The validation & test only contain plants with at least one
    non artificial image (unless all images are artificial).
    Artificial images are deleted from the test and validation mapping.

    Args:
        plant_mapping (pd.DataFrame): plant_id -> volume, plant_id -> images mapping
        rel_test_set_size (float, optional): Relative (to number of plants) test set size. Defaults to 0.2.
        rel_val_set_size (float, optional): Relative (to number of plants) validation set size. Defaults to 0.1.
        verbose (bool, optional): Print split sizes. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_mapping, validation_mapping, test_mapping

    Warnings:
        Warns if the plant mapping contains less than 50 plants with real world images.
        
    Examples:
        >>> plant_mapping
                volume               images artificial_mask
        2_6_1   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_2   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_3   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_4   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_5   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_6   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_7   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_8   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_9   5000.0  [2_6_1_0.jpg, arti]   [False, True]
        2_6_10  5000.0  [2_6_1_0.jpg, arti]   [False, True]
        >>> train, val, test = split_(plant_mapping=df_in,
        ...                           rel_test_set_size=0.2,
        ...                           rel_val_set_size=0.1,
        ...                           random_generator=np.random.default_rng(0))
        >>> val
               volume         images artificial_mask
        2_6_1  5000.0  [2_6_1_0.jpg]         [False]
    """
    
    n_plants = plant_mapping.shape[0]
    n_real_images = plant_mapping['artificial_mask'].apply(lambda x : len(x) - np.sum(x)).to_numpy()
    n_all_arti = np.sum(n_real_images == 0)
     
    # ----------------------------------------------------
    # Splitting the data into training and validation sets
    # ----------------------------------------------------
    if n_all_arti == n_plants:
        # all plants only contain artificial images plants
        warnings.warn("All plants artifical. Will validate exclusively on artifical images. min_view_train and min_view_test will be ignored.")

        train_size = n_plants - test_set_size - val_set_size
        shuffled_idx = np.arange(n_plants)
        random_generator.shuffle(shuffled_idx)
        train_indices , val_indices, test_indices = np.split(shuffled_idx, [train_size, train_size + val_set_size])

    else:
        # there is at least one plant with a real world image
        assert min_view_train <= min_view_test, "This does not really make sense"
        mask_usable_test = n_real_images >= min_view_test
        mask_usable_train = n_real_images >= min_view_train
        assert mask_usable_test.sum() >= test_set_size + val_set_size, f"Not enough usable plants for test set {mask_usable_test.sum()} vs {test_set_size + val_set_size}"

        usable_test_indices = np.nonzero(mask_usable_test)[0]
        np.random.shuffle(usable_test_indices)
        test_indices = usable_test_indices[:test_set_size]
        val_indices = usable_test_indices[test_set_size:test_set_size + val_set_size]
        mask_usable_train[test_indices] = False
        mask_usable_train[val_indices] = False
        train_indices = np.nonzero(mask_usable_train)
    
    train_mapping = plant_mapping.iloc[train_indices]
    test_mapping = plant_mapping.iloc[test_indices]
    val_mapping = plant_mapping.iloc[val_indices]
    
    assert np.intersect1d(train_mapping.index, val_mapping.index).shape[0] == 0
    assert np.intersect1d(train_mapping.index, test_mapping.index).shape[0] == 0
    assert np.intersect1d(val_mapping.index, test_mapping.index).shape[0] == 0
    
    
    # remove the artificial images from test and val
    if not (n_all_arti == n_plants): # not all plants exclusively artificial
        val_mapping = val_mapping.apply(remove_artificial, axis=1)
        test_mapping = test_mapping.apply(remove_artificial, axis=1)
    
    if verbose:
        print(f"Plants in mapping: {n_plants}.")
        print(f"Total train set size {train_mapping.shape[0]}")
        print(f"Total validation set size {val_mapping.shape[0]}")
        print(f"Total test set size {test_mapping.shape[0]}")

    return train_mapping, val_mapping, test_mapping

def compute_and_store_split(
        df_mapping : pd.DataFrame, 
        mapping_plants_path : Path,
        mapping_train_path : Path,
        mapping_val_path : Path, 
        mapping_test_path : Path,
        mapping_plants_val_path: Path,
        value_column : str,
        random_generator : np.random.Generator,
        test_set_size : int = 300,
        val_set_size : int = 150,
        min_view_train: int = 3,
        min_view_test: int = 6,
        verbose : bool = False,
        ):
    """Compute the train / val / test split and store the output
    in json format at the given locations.
    
    The output files will be overwritten if already exsitent.

    Args:
        mapping_image_path (DataFrame): Datafraem file containing the image to volume / plant mapping.
        mapping_plants_path (Path): json file path to store plant to volume / images mapping
        mapping_train_path (Path): json file to store subset of plants_mapping for training
        mapping_val_path (Path): json file to store subset of plants_mapping for validation
        mapping_test_path (Path): json file to store subset of plants_mapping for testing
        value_column (str): Column name containing the value.
        random_generator (Generator): Randomness to pull shuffling from.
    """
    
    plant_mapping = compute_plant_mapping_(df_mapping=df_mapping)

    train_mapping, val_mapping, test_mapping = split_(plant_mapping=plant_mapping,
                                                      test_set_size=test_set_size,
                                                      val_set_size=val_set_size,
                                                      min_view_train=min_view_train,
                                                      min_view_test=min_view_test,
                                                      random_generator=random_generator,
                                                      verbose=verbose)

    mapping_train_path.parent.mkdir(parents=False, exist_ok=True)
    train_mapping.to_json(mapping_train_path, orient='index')

    mapping_val_path.parent.mkdir(parents=False, exist_ok=True)
    val_mapping.to_json(mapping_val_path, orient='index')
    
    mapping_test_path.parent.mkdir(parents=False, exist_ok=True)
    test_mapping.to_json(mapping_test_path, orient='index')

    mapping_plants_val_path.parent.mkdir(parents=False, exist_ok=True)
    all_val = create_all_val([train_mapping, val_mapping, test_mapping], min_view_test)
    all_val.to_json(mapping_plants_val_path, orient='index')
    
    mapping_plants_path.parent.mkdir(parents=False, exist_ok=True)
    plant_mapping.to_json(mapping_plants_path, orient='index')



