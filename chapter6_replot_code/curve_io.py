import ast
import csv
import json
from pathlib import Path


def _load_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON list")
    return [float(x) for x in data]


def _load_txt(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    data = ast.literal_eval(text)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a Python list")
    return [float(x) for x in data]


def _load_csv(path: Path):
    values = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            values.append(float(row[0]))
    return values


def load_curve(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".json":
        data = _load_json(path)
    elif suffix == ".txt":
        data = _load_txt(path)
    elif suffix == ".csv":
        data = _load_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if len(data) < 2:
        raise ValueError(f"Curve in {path} is too short")
    return data
