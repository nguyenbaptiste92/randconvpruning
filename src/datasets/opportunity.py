import collections
import glob
import io
import os
import zipfile
from functools import reduce
from operator import add, and_, concat, or_
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from scipy import interpolate
from sklearn.model_selection import train_test_split

from .utils import dataset_func_chain

pd.set_option("display.max_rows", 10000)

zip_file_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip"
number_to_label_dict = {0: "null", 1: "stand", 2: "walk", 4: "sit", 5: "lie"}
label_dict = {"null": 0, "lie": 1, "sit": 2, "stand": 3, "walk": 4}

column_files = "column_names.txt"
column_list = [
    "MILLISEC",
    "Accelerometer RKN",
    "Accelerometer HIP",
    "Accelerometer LUA",
    "Accelerometer RUA",
    "Accelerometer LH",
    "Accelerometer BACK",
    "Accelerometer RWR",
    "Accelerometer LWR",
    "Accelerometer RH",
    "InertialMeasurementUnit BACK acc",
    "InertialMeasurementUnit RUA acc",
    "InertialMeasurementUnit RLA acc",
    "InertialMeasurementUnit LUA acc",
    "InertialMeasurementUnit LLA acc",
    "InertialMeasurementUnit L-SHOE Nav_A",
    "InertialMeasurementUnit L-SHOE Body_A",
    "InertialMeasurementUnit R-SHOE Nav_A",
    "InertialMeasurementUnit R-SHOE Body_A",
    "Locomotion",
]

localisation_dict = {
    "UA": ["RUA_", "LUA^", "LUA_", "RUA^", "RUA", "LUA"],
    "LA": ["LLA", "RLA"],
    "WR": ["LWR", "RWR"],
    "H": ["RH", "LH"],
    "BACK": ["BACK"],
    "HIP": ["HIP"],
    "KN": ["RKN_", "RKN^"],
    "SHOE": ["L-SHOE", "R-SHOE"],
}

###############################################################################################################################################
# Function to download Opportunity HAR dataset from "https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip"
# The following processing is realized: Sampling sample with a windows of 3 seconds, filter transition samples, filter incorrect values (NaN), re-sample to 50 Hz, assemble the dataset by position, normalize
###############################################################################################################################################


def get_column_names(data_dir, detailed=False):
    path = os.path.join(
        data_dir,
        "OpportunityUCIDataset",
        "dataset",
        column_files,
    )
    with open(path) as f:
        lines = f.readlines()
    lines = [line for line in lines if line.startswith("Column:")]
    lines = [line.split("\n")[0] for line in lines]
    if detailed:
        lines = [line.split(";") for line in lines]
        lines = [
            [line[0]] if len(line) == 1 else [line[0], line[1].split(",")[1]]
            for line in lines
        ]
        lines = [
            [line[0].split()[2:]]
            if len(line) == 1
            else [line[0].split()[2:], line[1].split()[2:]]
            for line in lines
        ]
        lines = [
            [" ".join(line[0])]
            if len(line) == 1
            else [" ".join(line[0]), " ".join(line[1])]
            for line in lines
        ]
        lines = [
            line[0] if len(line) == 1 else line[0] + " (" + line[1] + ")"
            for line in lines
        ]
    else:
        lines = [line.split(";")[0] for line in lines]
        lines = [line.split()[2:] for line in lines]
        lines = [" ".join(line) for line in lines]

    # Corrects error in column files
    # Correct triplet of names (for exemple Column: 166 Accelerometer MILK accX should be Column: 166 Accelerometer MILK accY)
    list_double = [
        item for item, count in collections.Counter(lines).items() if count > 1
    ]
    dict_double_index = {
        item: [i for i, x in enumerate(lines) if x == item]
        for item in list_double
    }
    assert reduce(
        and_,
        [len(elem) == 3 for elem in dict_double_index.values()],
    ), (
        "For this error, there should be triplet (for exemple Column: 165 Accelerometer MILK accX, Column: 166 Accelerometer MILK accX,  Column: 166 Accelerometer MILK accX)."
    )
    indexes = reduce(
        concat,
        [[elem[1], elem[2]] for elem in dict_double_index.values()],
    )
    replacements = reduce(
        concat,
        [
            [elem.replace("accX", "accY"), elem.replace("accX", "accZ")]
            for elem in dict_double_index
        ],
    )
    for index, replacement in zip(indexes, replacements, strict=False):
        lines[index] = replacement

    # Return columns
    return lines


"""
Arguments : -data_dir (string): directory path where the data will be saved
            -positional (bool): Save the dataset as multiple dataset by position or save it in one dataset
"""


def download_opportunity(data_dir, positional=False):
    # Check if pre-process already done
    if positional:
        pos_path = os.path.join(
            data_dir,
            "OpportunityUCIDataset",
            "dataset",
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
        if count == 8:
            print("Opportunity Positional Dataset already preprocessed")
            return dico_dataset

    else:
        glob_path = os.path.join(
            data_dir,
            "OpportunityUCIDataset",
            "dataset",
            "Globale",
        )
        if os.path.exists(
            os.path.join(glob_path, "input.npy"),
        ) and os.path.exists(os.path.join(glob_path, "label.npy")):
            print("Opportunity Globale Dataset already preprocessed")
            return {
                "global": {
                    "label": np.load(os.path.join(glob_path, "label.npy")),
                    "input": np.load(os.path.join(glob_path, "input.npy")),
                },
            }

    # Download from url
    if os.path.exists(os.path.join(data_dir, "OpportunityUCIDataset")):
        print("Opportunity dataset already downloaded.")
    else:
        print("Download Opportunity dataset.")
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(data_dir)

    # Extract data
    print("Extract Opportunity dataset.")
    columns = get_column_names(data_dir)
    path = os.path.join(data_dir, "OpportunityUCIDataset", "dataset")
    good_columns = [
        column
        for column in columns
        if reduce(or_, [column.startswith(elem) for elem in column_list])
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

        # Sample echantillons of 3 seconds (3*30 car la frequence est 30 Hz.)
        print("Sample dataset")
        df_list = [df.iloc[n : n + 90, :] for n in range(0, len(df), 90)]
        assert reduce(add, [len(elem) for elem in df_list]) == len(df), (
            "Split not done correctly."
        )
        df_list = [elem for elem in df_list if len(elem) == 90]
        print(len(df_list))

        # Filter transitions echantillons
        print("Filter Transition Echantillons")
        unique = [
            True if len(elem["Locomotion"].unique()) == 1 else False
            for elem in df_list
        ]
        filtered_df_list = [
            elem
            for elem, condition in zip(df_list, unique, strict=False)
            if condition == True
        ]
        print(len(filtered_df_list))

        # Separate by captor location then filter echantillon with NaN (besoin de rajouter des localisations et fusionner gauche et droite)
        print("Separate by accelerometer location")
        dico_location = {}
        for loc, captors in localisation_dict.items():
            dico_location[loc] = []
            for captor in captors:
                captor_columns = (
                    ["MILLISEC"]
                    + [elem for elem in good_columns if captor in elem]
                    + ["Locomotion"]
                )
                captor_list_df = [
                    elem[captor_columns] for elem in filtered_df_list
                ]
                dico_location[loc] += [
                    elem
                    for elem in captor_list_df
                    if not elem.isnull().values.any()
                ]

        # Resample to 50 Hz
        print("Resample to 50 Hz")

        def SampleTo50Hz(df):
            new_df = pd.DataFrame(index=range(150), columns=df.columns)
            new_df["MILLISEC"] = (
                np.linspace(0, 2980, 150) + df["MILLISEC"].iloc[0]
            )
            new_df["Locomotion"] = df["Locomotion"].iloc[0]
            new_df.iloc[:, 1] = interpolate.interp1d(
                df["MILLISEC"],
                df.iloc[:, 1],
                kind="linear",
                fill_value="extrapolate",
            )(new_df["MILLISEC"])
            new_df.iloc[:, 2] = interpolate.interp1d(
                df["MILLISEC"],
                df.iloc[:, 2],
                kind="linear",
                fill_value="extrapolate",
            )(new_df["MILLISEC"])
            new_df.iloc[:, 3] = interpolate.interp1d(
                df["MILLISEC"],
                df.iloc[:, 3],
                kind="linear",
                fill_value="extrapolate",
            )(new_df["MILLISEC"])
            return new_df

        for key, liste in dico_location.items():
            dico_location[key] = [SampleTo50Hz(elem) for elem in liste]

        # Assemble into a a dataset per location
        print("Assemble dataset")
        for key, liste in dico_location.items():
            if key not in dataset:
                dataset[key] = {"label": [], "input": []}
            if len(liste) > 0:
                label_liste = [elem["Locomotion"].iloc[0] for elem in liste]
                label_liste = [
                    number_to_label_dict[elem] for elem in label_liste
                ]
                label_liste = [label_dict[elem] for elem in label_liste]
                label_array = np.array(label_liste)
                dataset[key]["label"].append(label_array)
                input_liste = [
                    elem.iloc[:, [1, 2, 3]].to_numpy() for elem in liste
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
# Function to load the pre-process Opportunity dataset as tf.data.Dataset
###############################################################################################################################################

"""
Arguments : -path : path of the Opportunity dataset
            -proportion (float between 0 and 1): proportion for the train/test split
            -train (bool): take the train dataset
            -inter (bool): True, use other dataset for the domain generalization settings; False, use the other body position of Opportunity for the domain generalization settings
            -source_localisation (string): if inter==False, select the body localisation for the source domain ("Back","H"(hand),"HIP","KN"(knee),"LA"(lower arm),"SHOE","UA"(upper arm),"WR"(wrist))
            -transform (liste of function): apply transformations to the dataset
"""


def get_opportunity(
    path,
    train=True,
    inter=True,
    proportion=0.8,
    transform=[],
    source_localisation="H",
):
    transform_func = dataset_func_chain(transform)
    if inter:
        dataset = download_opportunity(path, positional=False)
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

    dataset = download_opportunity(path, positional=True)
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
