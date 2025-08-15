import os, yaml

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
