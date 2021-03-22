import os
import config
import shutil
import requests
import pandas as pd
from tensorflow.keras import utils


def data_preprocessing():
    if not os.path.exists(config.dataset_dir):
        dataset = config.dataset_url.split("/")[-1]
        with open(dataset, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
        dataset = "data.tar.gz"
        shutil.unpack_archive(dataset)

    train_df = pd.read_csv(config.train_dir, nrows=140000)
    valid_df = pd.read_csv(config.valid_dir)
    test_df = pd.read_csv(config.test_dir)

    train_df.dropna(axis=0, inplace=True)

    train_df = (
        train_df[train_df.similarity != "-"].sample(frac=1.0).reset_index(drop=True)
    )
    valid_df = (
        valid_df[valid_df.similarity != "-"].sample(frac=1.0).reset_index(drop=True)
    )

    train_df["label"] = train_df["similarity"].apply(
        lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
    )
    y_train = utils.to_categorical(train_df.label, num_classes=3)

    valid_df["label"] = valid_df["similarity"].apply(
        lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
    )
    y_valid = utils.to_categorical(valid_df.label, num_classes=3)

    test_df["label"] = test_df["similarity"].apply(
        lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
    )
    y_test = utils.to_categorical(test_df.label, num_classes=3)

    return train_df, valid_df, test_df, y_train, y_valid, y_test
