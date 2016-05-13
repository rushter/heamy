import hashlib

import numpy as np
import os
import json
import pandas as pd

CACHE_DIR = '.cache/heamy/'


class Cache(object):
    def __init__(self, hash, prefix=''):
        if prefix != '':
            hash = '%s%s' % (prefix, hash)
        self._hash = hash
        self._hash_dir = os.path.join(CACHE_DIR, hash)

    def store(self, key, data):
        """Takes an array and stores it in the cache."""
        if not os.path.exists(self._hash_dir):
            os.makedirs(self._hash_dir)

        if isinstance(data, pd.DataFrame):
            columns = data.columns.tolist()
            np.save(os.path.join(self._hash_dir, key), data.values)
            json.dump(columns, open(os.path.join(self._hash_dir, '%s.json' % key), 'w'))
        else:
            np.save(os.path.join(self._hash_dir, key), data)

    def retrieve(self, key):
        """Retrieves a cached array if possible."""
        columns_file = os.path.join(self._hash_dir, '%s.json' % key)
        cache_file = os.path.join(self._hash_dir, key) + '.npy'

        if os.path.exists(cache_file):
            data = np.load(cache_file)
            if os.path.exists(columns_file):
                columns = json.load(open(columns_file, 'r'))
                data = pd.DataFrame(data, columns=columns)
        else:
            return None

        return data

    @property
    def available(self):
        return os.path.exists(os.path.join(CACHE_DIR, self._hash))


def numpy_buffer(ndarray):
    """Creates a buffer from c_contiguous numpy ndarray."""
    # Credits to: https://github.com/joblib/joblib/blob/04b001861e1dd03a857b7b419c336de64e05714c/joblib/hashing.py

    if isinstance(ndarray, (pd.Series, pd.DataFrame)):
        ndarray = ndarray.values

    if ndarray.flags.c_contiguous:
        obj_c_contiguous = ndarray
    elif ndarray.flags.f_contiguous:
        obj_c_contiguous = ndarray.T
    else:
        obj_c_contiguous = ndarray.flatten()

    obj_c_contiguous = obj_c_contiguous.view(np.uint8)

    if hasattr(np, 'getbuffer'):
        return np.getbuffer(obj_c_contiguous)
    else:
        return memoryview(obj_c_contiguous)


def np_hash(x):
    m = hashlib.new('md5')
    m.update(numpy_buffer(x))
    return m.hexdigest()
