# -*- coding: utf-8 -*-
import re

from scipy.sparse import csr_matrix, coo_matrix, hstack, vstack
import numpy as np
import logging


class XGBParser(object):
    def __init__(self):
        self.groups = set()

    def load_dump(self, path):

        # model.dump_model('dump.raw.txt')
        dump = open(path, 'r').read()
        search = re.finditer('\[f([0-9]*?)<(.*?)\]', dump, re.MULTILINE)
        for group in search:
            group = group.groups()
            idx = int(group[0])
            val = float(group[1])
            self.groups.add((idx, val))
        logging.info('Found %s splits.' % (len(self.groups)))

    def transform(self, X):
        output = []
        for i, group in enumerate(self.groups):
            idx, val = group
            cond = csr_matrix((X[:, idx] < val).reshape((-1, 1))).astype(np.int8)
            output.append(cond)
        output = hstack(output, dtype=np.int8)
        return output
