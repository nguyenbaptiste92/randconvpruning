import glob
import io
import os
import zipfile
from functools import reduce
from operator import add
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from scipy import interpolate
from sklearn.model_selection import train_test_split

from .utils import dataset_func_chain

pd.set_option("display.max_rows", 10000)

zip_file_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"

IMU_column = [
    "Temperature",
    "Accelerometer_1 X",
    "Accelerometer_1 Y",
    "Accelerometer_1 Z",
    "Accelerometer_2 X",
    "Accelerometer_2 Y",
    "Accelerometer_2 Z",
    "Gyroscope X",
    "Gyroscope Y",
    "Gyroscope Z",
    "Magnetometer X",
    "Magnetometer Y",
    "Magnetometer Z",
    "Orientation 1",
    "Orientation 2",
    "Orientation 3",
    "Orientation 4",
]
columns = (
    ["Timestamp", "Activity_ID", "Heart Rate"]
    + ["IMU hand " + string for string in IMU_column]
    + ["IMU chest " + string for string in IMU_column]
    + ["IMU ankle " + string for string in IMU_column]
)

number_to_label_dict = {
    0: "null",
    1: "lie",
    2: "sit",
    3: "stand",
    4: "walk",
    5: "run",
    6: "bike",
    7: "nordic_walking",
    12: "stairsup",
    13: "stairsdown",
    16: "vacuum",
    17: "iron",
    24: "rope_jumping",
}
label_dict = {
    "null": 0,
    "lie": 1,
    "sit": 2,
    "stand": 3,
    "walk": 4,
    "run": 5,
    "bike": 6,
    "stairsup": 7,
    "stairsdown": 8,
    "nordic_walking": 10,
    "vacuum": 11,
    "iron": 12,
    "rope_jumping": 13,
}

###############################################################################################################################################
# Function to download PAMAP2 HAR dataset from "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
# The following processing is realized: Sampling sample with a windows of 3 seconds, filter transition samples, filter incorrect values (NaN), re-sample to 50 Hz, assemble the dataset by position, normalize
###############################################################################################################################################

"""
Arguments : -data_dir (string): directory path where the data will be saved
            -positional (bool): Save the dataset as multiple dataset by position or save it in one dataset
"""


def download_PAMAP2(data_dir, positional=False):
    # Check if pre-process already done
    if positional:
        pos_path = os.path.join(
            data_dir,
            "PAMAP2_Dataset",
            "Protocol",
            "Positionale",
        )
        folder = glob.glob(pos_path + os.sep + "*")
        count = 0
        dico_dataset = {}
        for fold in folder:
            if os.path.exists(
                os.path.join(fold, "input.npy"),
            ) and os.path.exists(os.path.join(fold, "label.npy")):
                name = os.path.basename(os.path.normpath(fold))
                dico_dataset[name] = {
                    "label": np.load(os.path.join(fold, "label.npy")),
                    "input": np.load(os.path.join(fold, "input.npy")),
                }
                count += 1
        if count == 3:
            print("PAMAP2 Positional Dataset already preprocessed")
            return dico_dataset

    else:
        glob_path = os.path.join(
            data_dir,
            "PAMAP2_Dataset",
            "Protocol",
            "Globale",
        )
        if os.path.exists(
            os.path.join(glob_path, "input.npy"),
        ) and os.path.exists(os.path.join(glob_path, "label.npy")):
            print("PAMAP2 Globale Dataset already preprocessed")
            return {
                "global": {
                    "label": np.load(os.path.join(glob_path, "label.npy")),
                    "input": np.load(os.path.join(glob_path, "input.npy")),
                },
            }

    # Download from url
    if os.path.exists(os.path.join(data_dir, "PAMAP2_Dataset")):
        print("PAMAP2 dataset already downloaded.")
    else:
        print("Download PAMAP2 dataset.")
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(data_dir)

    # Extract data
    print("Extract PAMAP2 dataset.")
    path = os.path.join(data_dir, "PAMAP2_Dataset", "Protocol")
    good_columns = ["Timestamp", "Activity_ID"] + [
        column for column in columns if "Accelerometer_1" in column
    ]

    dataset = {}
    for filepath in glob.glob(path + os.sep + "*.dat"):
        print("Read ", filepath)
        df = pd.read_csv(
            filepath,
            header=None,
            delim_whitespace=True,
            error_bad_lines=False,
            names=columns,
        )
        df = df[good_columns]

        # Sample echantillons of 3 seconds (3*100 car la frequence est 100 Hz.)
        print("Sample dataset")
        df_list = [df.iloc[n : n + 300, :] for n in range(0, len(df), 300)]
        assert reduce(add, [len(elem) for elem in df_list]) == len(df), (
            "Split not done correctly."
        )
        df_list = [elem for elem in df_list if len(elem) == 300]
        print(len(df_list))

        # Filter transitions echantillons
        print("Filter Transition Echantillons")
        unique = [
            True if len(elem["Activity_ID"].unique()) == 1 else False
            for elem in df_list
        ]
        filtered_df_list = [
            elem
            for elem, condition in zip(df_list, unique, strict=False)
            if condition == True
        ]
        print(len(filtered_df_list))

        # Filter activity (we keep only the activity with an Id of 1,2,3,4,5,6,7,12,13,16,17,24
        print("Filter Activity")
        filtered_df_list = [
            elem
            for elem in filtered_df_list
            if elem["Activity_ID"].iloc[0]
            in [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
        ]
        print(len(filtered_df_list))

        # Separate by captor location then filter echantillon with NaN
        print("Separate by accelerometer location")
        locations = list(
            set(
                [
                    elem.split()[1]
                    for elem in good_columns
                    if "Accelerometer" in elem
                ],
            ),
        )
        dico_location = {}
        for loc in locations:
            loc_columns = ["Timestamp", "Activity_ID"] + [
                elem for elem in good_columns if loc in elem
            ]
            dico_location[loc] = [
                elem[loc_columns] for elem in filtered_df_list
            ]
            dico_location[loc] = [
                elem
                for elem in dico_location[loc]
                if not elem.isnull().values.any()
            ]

        # Resample to 50 Hz
        print("Resample to 50 Hz")

        def SampleTo50Hz(df):
            new_df = pd.DataFrame(index=range(150), columns=df.columns)
            new_df["Timestamp"] = (
                np.linspace(0, 2980, 150) + df["Timestamp"].iloc[0]
            )
            new_df["Activity_ID"] = df["Activity_ID"].iloc[0]
            new_df.iloc[:, 2] = interpolate.interp1d(
                df["Timestamp"],
                df.iloc[:, 2],
                kind="linear",
                fill_value="extrapolate",
            )(new_df["Timestamp"])
            new_df.iloc[:, 3] = interpolate.interp1d(
                df["Timestamp"],
                df.iloc[:, 3],
                kind="linear",
                fill_value="extrapolate",
            )(new_df["Timestamp"])
            new_df.iloc[:, 4] = interpolate.interp1d(
                df["Timestamp"],
                df.iloc[:, 4],
                kind="linear",
                fill_value="extrapolate",
            )(new_df["Timestamp"])
            return new_df

        for key, liste in dico_location.items():
            dico_location[key] = [SampleTo50Hz(elem) for elem in liste]

        # Assemble into a a dataset per location
        print("Assemble dataset")
        for key, liste in dico_location.items():
            if key not in dataset:
                dataset[key] = {"label": [], "input": []}
            if len(liste) > 0:
                label_liste = [elem["Activity_ID"].iloc[0] for elem in liste]
                label_liste = [
                    number_to_label_dict[elem] for elem in label_liste
                ]
                label_liste = [label_dict[elem] for elem in label_liste]
                label_array = np.array(label_liste)
                dataset[key]["label"].append(label_array)
                input_liste = [
                    elem.iloc[:, [2, 3, 4]].to_numpy() for elem in liste
                ]
                input_array = np.stack(input_liste)
                dataset[key]["input"].append(input_array)

    # Concatenate and linear transform to put data between -1 and 1
    print("Save processed dataset")
    final_dataset = {}
    for key, dico in dataset.items():
        final_dataset[key] = {}
        final_dataset[key]["label"] = np.concatenate(dico["label"])
        final_dataset[key]["input"] = np.concatenate(dico["input"])
        max_input, min_input = (
            np.max(final_dataset[key]["input"]),
            np.min(final_dataset[key]["input"]),
        )
        final_dataset[key]["input"] = (
            2
            * (
                (final_dataset[key]["input"] - min_input)
                / (max_input - min_input)
            )
            - 1
        )
        print(
            final_dataset[key]["label"].shape,
            final_dataset[key]["input"].shape,
        )

    # Save
    if positional:
        for key, dico in final_dataset.items():
            save_path = os.path.join(path, "Positionale", key)
            Path(save_path).mkdir(parents=True, exist_ok=True)
            np.save(
                save_path + os.sep + "label.npy",
                final_dataset[key]["label"],
            )
            np.save(
                save_path + os.sep + "input.npy",
                final_dataset[key]["input"],
            )
        return final_dataset
    liste_label = []
    liste_input = []
    for key, dico in final_dataset.items():
        liste_label.append(final_dataset[key]["label"])
        liste_input.append(final_dataset[key]["input"])
    label_array = np.concatenate(liste_label)
    input_array = np.concatenate(liste_input)
    save_path = os.path.join(path, "Globale")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    np.save(save_path + os.sep + "label.npy", label_array)
    np.save(save_path + os.sep + "input.npy", input_array)
    return {"global": {"label": label_array, "input": input_array}}


###############################################################################################################################################
# Function to load the pre-process PAMAP2 dataset as tf.data.Dataset
###############################################################################################################################################

"""
Arguments : -path : path of the PAMAP2 dataset
            -proportion (float between 0 and 1): proportion for the train/test split
            -train (bool): take the train dataset
            -inter (bool): True, use other dataset for the domain generalization settings; False, use the other body position of PAMAP2 for the domain generalization settings
            -source_localisation (string): if inter==False, select the body localisation for the source domain ("ankle","chest","hand")
            -transform (liste of function): apply transformations to the dataset
"""


def get_PAMAP2(
    path,
    train=True,
    inter=True,
    proportion=0.8,
    transform=[],
    source_localisation="ankle",
):
    transform_func = dataset_func_chain(transform)
    if inter:
        dataset = download_PAMAP2(path, positional=False)
        x = dataset["global"]["input"]
        y = dataset["global"]["label"]
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=proportion,
            random_state=1,
        )
        if train:
            dataset = tf.data.Dataset.from_tensor_slices(
                {"image": x_train, "label": y_train},
            )
            dataset = dataset.apply(
                tf.data.experimental.assert_cardinality(len(x_train)),
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                {"image": x_test, "label": y_test},
            )
            dataset = dataset.apply(
                tf.data.experimental.assert_cardinality(len(x_test)),
            )
        dataset = dataset.map(transform_func)
        return dataset

    dataset = download_PAMAP2(path, positional=True)
    x = dataset[source_localisation]["input"]
    y = dataset[source_localisation]["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=proportion,
        random_state=1,
    )
    if train:
        dt = tf.data.Dataset.from_tensor_slices(
            {"image": x_train, "label": y_train},
        )
        dt = dt.apply(tf.data.experimental.assert_cardinality(len(x_train)))
        dt = dt.map(transform_func)
        return dt
    dict_dataset = {}
    dt = tf.data.Dataset.from_tensor_slices({"image": x_test, "label": y_test})
    dt = dt.apply(tf.data.experimental.assert_cardinality(len(x_test)))
    dt = dt.map(transform_func)
    dict_dataset[source_localisation] = dt

    for key in dataset.keys():
        if key != source_localisation:
            x = dataset[key]["input"]
            y = dataset[key]["label"]
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                train_size=proportion,
                random_state=1,
            )
            dt = tf.data.Dataset.from_tensor_slices(
                {"image": x_test, "label": y_test},
            )
            dt = dt.apply(tf.data.experimental.assert_cardinality(len(x_test)))
            dt = dt.map(transform_func)
            dict_dataset[key] = dt
    return dict_dataset
