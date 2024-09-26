import os
import json
import yaml
import pickle
import pandas as pd
from typing import Any, List
import pyarrow.feather as feather
from scipy.sparse import csr_matrix
from datetime import datetime, timezone


def to_json(obj: dict[str, Any], path: str, encoding: str = "utf-8") -> None:
    """Write a json file."""
    make_folder(path, ["json"])
    with open(path, "w", encoding=encoding) as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def read_json(path: str, encoding: str = "utf-8") -> dict:
    """Read a json file."""
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def read_pickle(path: str) -> dict:
    """read pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)


def read_yaml(path: str, encoding: str = "utf-8") -> dict:
    """Read a yaml file."""
    with open(path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def to_pickle(obj: dict, path: str) -> None:
    """write pickle file"""
    make_folder(path, ["pkl", "pickle"])
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def make_folder(path, extension):
    dir_, file = os.path.split(path)
    if file.split(".")[-1] not in extension:
        raise ValueError(f"extension is not {extension}")
    if dir_ != "":
        os.makedirs(dir_, exist_ok=True)


def read_jsonl(path: str, encoding: str = "utf-8") -> List[dict]:
    """Read a jsonl file."""
    with open(path, "r", encoding=encoding) as f:
        return [json.loads(line.strip()) for line in f]


def to_jsonl(obj_list: List[dict], path: str, encoding: str = "utf-8") -> None:
    """Write a jsonl file."""
    make_folder(path, ["jsonl"])
    with open(path, "w", encoding=encoding) as f:
        for obj in obj_list:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def convert_unix_timestamp_to_utc(timestamp):
    # Convert milliseconds to seconds
    timestamp_seconds = timestamp

    # Convert to UTC time
    utc_time = datetime.fromtimestamp(timestamp_seconds, timezone.utc)

    # Format the output
    utc_formatted = utc_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    return utc_formatted


# def read_sparse_data(filename):
#     dense_df = pd.read_feather(f"{filename}_data.feather")
#     index_names = pd.read_feather(f"{filename}_index.feather")["index_names"]
#     column_names = pd.read_feather(f"{filename}_columns.feather")["column_names"]

#     coo = csr_matrix((dense_df["data"], (dense_df["row"], dense_df["col"])))

#     sparse_df = pd.DataFrame.sparse.from_spmatrix(
#         coo, index=index_names, columns=column_names
#     )

#     return sparse_df


def read_sparse_data(filename):
    dense_df = pd.read_feather(f"{filename}_data.feather")
    index_names = pd.read_feather(f"{filename}_index.feather")["index_names"]
    column_names = pd.read_feather(f"{filename}_columns.feather")["column_names"]

    coo = csr_matrix((dense_df["data"], (dense_df["row"], dense_df["col"])))

    # 실제 데이터의 크기 확인
    actual_rows = coo.shape[0]
    actual_cols = coo.shape[1]

    # 인덱스와 컬럼 이름의 길이 조정
    index_names = index_names[:actual_rows]
    column_names = column_names[:actual_cols]

    # sparse DataFrame 생성
    sparse_df = pd.DataFrame.sparse.from_spmatrix(coo)
    sparse_df.index = index_names
    sparse_df.columns = column_names

    return sparse_df


def to_sparse_data(sparse_df, filename):
    coo = sparse_df.sparse.to_coo()

    dense_df = pd.DataFrame({"data": coo.data, "row": coo.row, "col": coo.col})

    index_names = pd.Series(sparse_df.index.values, name="index_names")
    column_names = pd.Series(sparse_df.columns.values, name="column_names")

    feather.write_feather(dense_df, f"{filename}_data.feather")
    feather.write_feather(pd.DataFrame(index_names), f"{filename}_index.feather")
    feather.write_feather(pd.DataFrame(column_names), f"{filename}_columns.feather")

def evaluation(title,given_recommendation):
    NDCG =0
    HT = 0
    for i in range(len(given_recommendation)):
        if given_recommendation[i]== title:
            rank = i
    if rank<10:
        NDCG +=1