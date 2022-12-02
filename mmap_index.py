import pickle
import array
import mmap
import json


class Qry:

    def __init__(self, filepref):
        with open(filepref+".meta", "rt") as f:
            self.meta = json.load(f)
        self.data = open(filepref+".data", "rb")
        self.data_mmap = mmap.mmap(
            self.data.fileno(), 0, access=mmap.ACCESS_READ)

        self.index = array.array("L")
        with open(filepref+".index", "rb") as f:
            self.index.fromfile(f, self.meta["len"])

        self.lengths = array.array("I")
        with open(filepref+".lengths", "rb") as f:
            self.lengths.fromfile(f, self.meta["len"])

    def get(self, i):
        idx = self.index[i]
        length = self.lengths[i]
        self.data_mmap.seek(idx)
        data = self.data_mmap.read(length)
        return pickle.loads(data)
