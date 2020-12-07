import os
# from functools import lru_cache

import numpy as np
import pandas as pd

from ensalada import DATA_PATH, RATS, SESSION_TYPES

MAZE_SCALE_FACTOR = 4.2  # pixels per cm


class Session:

    def __init__(self, rat, session_type='CDEF', day=1, bin_size=100):

        assert rat in RATS, "Invalid rat name!"
        assert session_type in SESSION_TYPES, "Invalid session type!"

        self.rat = rat
        self.session_type = session_type
        self.day = day
        self.data_path = os.path.join(DATA_PATH, rat, session_type + str(day))

        self.bin_size = bin_size

    @property
    def data_path(self):
        return self._data_path

    @data_path.setter
    def data_path(self, path):
        assert os.path.isdir(path), "Invalid data path!"
        self._data_path = path

    @property
    def _spikes_path(self):
        return os.path.join(self.data_path, "spikes.h5")

    @property
    def spikes_file(self):
        return pd.read_hdf(self._spikes_path)

    @property
    def _lfp_path(self):
        return os.path.join(self.data_path, "lfp.h5")

    @property
    def lfp_file(self):
        return pd.read_hdf(self._lfp_path)

    @property
    def _events_path(self):
        return os.path.join(self.data_path, "events.h5")

    @property
    def events_file(self):
        return pd.read_hdf(self._events_path)

    @property
    def _position_path(self):
        return os.path.join(self.data_path, "position.h5")

    @property
    def position_file(self):
        return pd.read_hdf(self._position_path)

    @property
    def num_units(self):
        return len(self.spikes_file)

    @property
    def num_trials(self):
        pass

    def position(self, bin_size=100):
        def _center_and_scale(a):
            a = a - np.mean([np.nanpercentile(a, p) for p in [5, 95]])
            return a / MAZE_SCALE_FACTOR

        pos = self.position_file.apply(_center_and_scale, axis=0)
        pos.index = pd.to_datetime(pos.index.values, unit='s')

        return pos.resample(f"{str(self.bin_size)}ms").apply(np.nanmean)

    def velocity(self):
        """Calculate from the derivative of the binned position"""
        pos = self.position()

        pass

    def detect_interneurons(self):
        pass



    def find_immobile_periods(self, threshold=5):
        "Find when the animal is not moving"
        pass

    def find_iti_periods(self):
        "Find when the animal is in the ITI chamber"
        pass

    def labels(self):
        "Trial labels dataframe"
        pass

    def rasters(self):
        "Cells x trials spike count dataframe"
        pass

    def format_online_data(self):
        """
        Returns
        -------
        X : array (num_samples, num_units)
            'Flattened' samples matrix, for model fitting.
        y : array (num_samples, num_labels)
            Indicator variables for each label
        t : array (num_samples,)
            Integer trial number for each sample
        """
        pass

    def format_offline_data(self):
        """
        Returns
        -------
        X : array (num_samples, num_units)
            'Flattened' samples matrix, for model fitting.
        y :


        """
        pass


class Rat(list):

    def __init__(self, rat):

        assert rat in RATS, "Invalid rat name!"

        # init a session object for all sessions and init list
