import pandas as pd
import json

def read_file(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format")
