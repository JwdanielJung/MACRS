import os
import pandas as pd
from src.utils import read_json


def load_data(path="/data/crs/movielens/final/240911"):
    train = pd.read_feather(os.path.join(path, "train.feather"))
    train["difficulty"] = [
        read_json(os.path.join(path, f"persona/train_{idx}.json"))["difficulty"]
        for idx in train.index
    ]
    dev = pd.read_feather(os.path.join(path, "dev.feather"))
    test = pd.read_feather(os.path.join(path, "test.feather"))
    return {"train": train, "dev": dev, "test": test}
