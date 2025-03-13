import glob
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from scipy import interpolate
from sklearn.model_selection import train_test_split

from .utils import dataset_func_chain

pd.set_option("display.max_rows", 1000)

zip_file_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity recognition exp.zip"
column = ["Creation_Time", "x", "y", "z", "gt"]
label_dict = {
    "null": 0,
    "sit": 2,
    "stand": 3,
    "walk": 4,
    "bike": 6,
    "stairsup": 7,
    "stairsdown": 8,
}

###############################################################################################################################################
# Function to download HHAR dataset from "https://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity recognition exp.zip"
# The following processing is realized: Sampling sample with a windows of 3 seconds, filter transition samples, filter incorrect values (NaN), re-sample to 50 Hz, assemble the dataset by device, normalize
###############################################################################################################################################

"""
Arguments : -data_dir (string): directory path where the data will be saved
            -device (bool): Save the dataset as multiple dataset by device or save it in one dataset
"""


def download_HHAR(data_dir, device=False):
    path = os.path.join(data_dir, "Activity recognition exp")

    # Check if pre-process already done
    if device:
        pos_path = os.path.join(path, "Device")
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
        if count == 6:
            print("HHAR Device Dataset already preprocessed")
            return dico_dataset

    else:
        glob_path = os.path.join(path, "Globale")
        if os.path.exists(
            os.path.join(glob_path, "input.npy"),
        ) and os.path.exists(os.path.join(glob_path, "label.npy")):
            print("HHAR Globale Dataset already preprocessed")
            return {
                "global": {
                    "label": np.load(os.path.join(glob_path, "label.npy")),
                    "input": np.load(os.path.join(glob_path, "input.npy")),
                },
            }

    # Download from url
    if os.path.exists(os.path.join(data_dir, "Activity recognition exp")):
        print("HHAR dataset already downloaded.")
    else:
        print("Download HHAR dataset.")
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(data_dir)

    print("Extract HHAR dataset.")
    # good_columns = ["Timestamp","Activity_ID"]+[column for column in columns if "Accelerometer_1" in column]

    dataset = {}

    for filepath in glob.glob(path + os.sep + "*_accelerometer.csv"):
        print(filepath)
        dataframe = pd.read_csv(filepath)

        gb = dataframe.groupby(["User", "Model", "Device"])
        dico = {x: gb.get_group(x)[column] for x in gb.groups}

        # Sample echantillons of 3 seconds
        dico_liste_sample = {}
        for key, df in dico.items():
            df["sample"] = pd.to_datetime(df["Creation_Time"]).dt.floor("3S")
            gb = df.groupby(["sample"])
            dico_liste_sample[key] = [
                gb.get_group(x)[column] for x in gb.groups
            ]

        # Filter Nan Echantillon,Transition Echantillon  et Echantillon with few value
        for key in dico_liste_sample:
            # print(key)
            dico_liste_sample[key] = [
                elem
                for elem in dico_liste_sample[key]
                if not elem.isnull().values.any()
            ]
            dico_liste_sample[key] = [
                elem.drop_duplicates(
                    subset="Creation_Time",
                    keep="first",
                    inplace=False,
                )
                for elem in dico_liste_sample[key]
            ]
            dico_liste_sample[key] = [
                elem
                for elem in dico_liste_sample[key]
                if not len(elem["gt"].unique()) > 1
            ]
            len_liste = len(dico_liste_sample[key])
            if len_liste > 0:
                max_lenght = max([len(elem) for elem in dico_liste_sample[key]])
            dico_liste_sample[key] = [
                elem
                for elem in dico_liste_sample[key]
                if (len(elem) > 0.8 * max_lenght and len(elem) >= 40)
            ]
            # print(len(dico_liste_sample[key]))

        # Resample to 50 Hz
        print("Resample to 50 Hz")

        def SampleTo50Hz(df):
            new_df = pd.DataFrame(index=range(150), columns=df.columns)
            new_df["Creation_Time"] = (
                np.linspace(0, 3000000000, 150) + df["Creation_Time"].iloc[0]
            )
            new_df["gt"] = df["gt"].iloc[0]
            new_df.iloc[:, 1] = interpolate.interp1d(
                df["Creation_Time"],
                df.iloc[:, 1],
                kind="linear",
                fill_value="extrapolate",
            )(new_df["Creation_Time"])
            new_df.iloc[:, 2] = interpolate.interp1d(
                df["Creation_Time"],
                df.iloc[:, 2],
                kind="linear",
                fill_value="extrapolate",
            )(new_df["Creation_Time"])
            new_df.iloc[:, 3] = interpolate.interp1d(
                df["Creation_Time"],
                df.iloc[:, 3],
                kind="linear",
                fill_value="extrapolate",
            )(new_df["Creation_Time"])
            return new_df

        for key in dico_liste_sample:
            dico_liste_sample[key] = [
                SampleTo50Hz(elem) for elem in dico_liste_sample[key]
            ]

        # Assemble into a a dataset
        for key in dico_liste_sample:
            if len(dico_liste_sample[key]) > 0:
                input_array = np.stack(
                    [
                        elem.iloc[:, [1, 2, 3]].to_numpy()
                        for elem in dico_liste_sample[key]
                    ],
                )
                label_array = np.array(
                    [
                        label_dict[elem.iloc[0, 4]]
                        for elem in dico_liste_sample[key]
                    ],
                )

                # Add to position
                if key[1] not in dataset:
                    dataset[key[1]] = {"label": [], "input": []}
                dataset[key[1]]["label"].append(label_array)
                dataset[key[1]]["input"].append(input_array)

    # Concatenate and linear transform to put data between -1 and 1
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
            key,
            final_dataset[key]["label"].shape,
            final_dataset[key]["input"].shape,
        )

    # Save
    if device:
        for key, dico in final_dataset.items():
            save_path = os.path.join(path, "Device", key)
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
# Function to load the pre-process Opportunity dataset as tf.data.Dataset
###############################################################################################################################################

"""
Arguments : -path : path of the HHAR dataset
            -proportion (float between 0 and 1): proportion for the train/test split
            -train (bool): take the train dataset
            -inter (bool): True, use other dataset for the domain generalization settings; False, use the otherdevice of HHAR for the domain generalization settings
            -source_localisation (string): if inter==False, select the devive for the source domain ("gear","lgwatch","nexus4","s3","s3mini","samsungold")
            -transform (liste of function): apply transformations to the dataset
"""


def get_HHAR(
    path,
    train=True,
    inter=True,
    proportion=0.8,
    transform=[],
    source_localisation="s3",
):
    transform_func = dataset_func_chain(transform)
    if inter:
        dataset = download_HHAR(path, device=False)
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

    dataset = download_HHAR(path, device=True)
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

    for key in dataset:
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
