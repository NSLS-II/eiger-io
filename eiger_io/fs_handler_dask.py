import dask.array as da
from pims import FramesSequence
from databroker.assets.handlers_base import HandlerBase
import re
import h5py

import os


class PIMSDask(FramesSequence):
    ''' the dask version of PIMS, takes dask array.

        Notes
        -----
            - this should eventually become a PR into PIMS.
            - this should be upgraded to allow nested FramesSequences
                (need to allow for defining axes etc)
    '''
    # the regexp patterns for expected files
    # here it is just file containing "master" but could potentially be
    # expanded upon
    def __init__(self, data, md=None):
        '''
            Initialized a lazy loader for EigerImages
            Parameters
            ----------
            data : dask.array.Array
                the data
            md : dict, optional
                the dictionary of metadata
        '''
        self._data = data
        self._md = md

    @property
    def md(self):
        return self._md

    def get_frame(self, i):
        return self._data[i].compute()

    def __len__(self):
        return len(self._data)

    @property
    def frame_shape(self):
        return self[0].shape

    @property
    def pixel_type(self):
        return self[0].dtype

    @property
    def dtype(self):
        return self.pixel_type

    @property
    def shape(self):
        return self.frame_shape

    def _to_dask(self):
        return self._data


class EigerDaskHandler(HandlerBase):
    EIGER_MD_LAYOUT = {
        'y_pixel_size': 'entry/instrument/detector/y_pixel_size',
        'x_pixel_size': 'entry/instrument/detector/x_pixel_size',
        'detector_distance': 'entry/instrument/detector/detector_distance',
        'incident_wavelength': 'entry/instrument/beam/incident_wavelength',
        'frame_time': 'entry/instrument/detector/frame_time',
        'beam_center_x': 'entry/instrument/detector/beam_center_x',
        'beam_center_y': 'entry/instrument/detector/beam_center_y',
        'count_time': 'entry/instrument/detector/count_time',
        'pixel_mask': 'entry/instrument/detector/detectorSpecific/pixel_mask',
    }
    specs = {'AD_EIGER2', 'AD_EIGER'}
    pattern = re.compile('(.*)master.*')
    def __init__(self, fpath, images_per_file):
        self.images_per_file = images_per_file
        self._base_path = fpath

    # this is on a per event level
    def __call__(self, seq_id):
        master_path = '{}_{}_master.h5'.format(self._base_path, seq_id)
        # check that 'master' is in file
        m = self.pattern.match(os.path.basename(master_path))

        if m is None:
            errormsg = "This reader expects filenames containing "
            errormsg += "the word 'master'. If the file was renamed, "
            errormsg += "revert to the original name given by the "
            errormsg += "detector."
            errormsg += "Got filename: {}".format(master_path)
            raise ValueError(errormsg)

        self._handle = h5py.File(master_path, 'r')
        try:
            self._entry = self._handle['entry']['data']  # Eiger firmware v1.3.0 and onwards
        except KeyError:
            self._entry = self._handle['entry']          # Older firmwares

        # TODO : perhaps remove the metadata eventually
        md = dict()
        with h5py.File(master_path, 'r') as f:
            md = {k: f[v].value for k, v in self.EIGER_MD_LAYOUT.items()}
        # the pixel mask from the eiger contains:
        # 1  -- gap
        # 2  -- dead
        # 4  -- under-responsive
        # 8  -- over-responsive
        # 16 -- noisy
        pixel_mask = md['pixel_mask']
        md['binary_mask'] = (md['pixel_mask'] == 0)
        md['framerate'] = 1./md['frame_time']

        # TODO : Return a multi-dimensional PIMS seq.
        # this is the logic that creates the linked dask array
        elements = list()
        key_names = sorted(list(self._entry.keys()))
        for keyname in key_names:
            print(f"{keyname}")
            val = self._entry[keyname]
            elements.append(da.from_array(val, chunks=val.chunks))

        # PIMS subclass using Dask
        return PIMSDask(da.concatenate(elements), md=md)
