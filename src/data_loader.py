"""Load train/test/OOT datasets and split into X/y."""
import pandas as pd
from . import config


def load_split(path):
    df = pd.read_csv(path)
    drop_cols = [c for c in config.ID_COLS if c in df.columns]
    y = df[config.TARGET].astype(int)
    X = df.drop(columns=drop_cols + [config.TARGET], errors="ignore")
    return X, y


def load_all():
    X_train, y_train = load_split(config.TRAIN_FILE)
    X_test, y_test = load_split(config.TEST_FILE)
    X_oot, y_oot = load_split(config.OOT_FILE)
    return (X_train, y_train), (X_test, y_test), (X_oot, y_oot)
