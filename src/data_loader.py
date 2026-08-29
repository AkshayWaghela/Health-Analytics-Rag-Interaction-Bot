from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import faiss


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def find_file(filename):
    """
    Find a file inside the data directory.
    """

    matches = list(DATA_DIR.rglob(filename))

    if not matches:
        return None

    return matches[0]


def load_dataframe():

    path = find_file("df_user.csv")

    if path is None:
        raise FileNotFoundError(
            "health_data.csv not found inside data/"
        )

    df = pd.read_csv(path)

    return df.reset_index(drop=True)


def load_documents():

    path = find_file("user_documents.pkl")

    if path is None:
        raise FileNotFoundError(
            "user_documents.pkl not found inside data/"
        )

    with open(path, "rb") as f:
        return pickle.load(f)


def load_embeddings():

    path = find_file("user_embeddings.npy")

    if path is None:
        raise FileNotFoundError(
            "user_embeddings.npy not found inside data/"
        )

    return np.load(path)


def load_faiss_index():

    path = find_file("user_faiss.index")

    if path is None:
        raise FileNotFoundError(
            "user_faiss.index not found inside data/"
        )

    return faiss.read_index(str(path))


def load_metadata():

    path = find_file("metadata.json")

    if path is None:
        return {}

    with open(path, "r") as f:
        return json.load(f)


def load_all_data():

    df = load_dataframe()

    documents = load_documents()

    embeddings = load_embeddings()

    index = load_faiss_index()

    metadata = load_metadata()

    return {
        "df": df,
        "documents": documents,
        "embeddings": embeddings,
        "index": index,
        "metadata": metadata
    }
