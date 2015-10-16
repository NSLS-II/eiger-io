from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

import numpy as np

from filestore.retrieve import HandlerBase
from .pims_readers import EigerImages


class EigerHandler(HandlerBase):
    specs = {'AD_EIGER'} | HandlerBase.specs

    def __init__(self, fpath, frame_per_point):
        # create pims handler
        self._base_path = fpath
        self.fpp = frame_per_point

    def __call__(self, seq_id):
        master_path = '{}_{}_master.h5'.format(self._base_path, seq_id)
        # TODO Return a multi-dimensional PIMS seq.
        return np.array(EigerImages(master_path))
