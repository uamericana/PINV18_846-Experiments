import os
import pathlib
import pickle

RUN = 0
GEN = 0

from itertools import groupby


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def checkpoint_file(checkpoint_dir, generation):
    return os.path.join(checkpoint_dir, f"gen_{generation:03d}.pkl")


def last_checkpoint(checkpoint_dir):
    files = list(pathlib.Path(checkpoint_dir).glob("**/gen_*.pkl"))
    files.sort()
    try:
        return str(files[-1])
    except:
        return None


def restore_datalog(experiment_dir):
    file = os.path.join(experiment_dir, "datalog.pkl")
    try:
        with open(file, "rb") as f:
            return pickle.load(f)
    except:
        pass
    return {}


def save_datalog(experiment_dir, datalog):
    file = os.path.join(experiment_dir, "datalog.pkl")
    with open(file, "wb") as f:
        pickle.dump(datalog, f)
