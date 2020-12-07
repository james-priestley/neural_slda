import os
# from functools import lru_cache

import numpy as np
import pandas as pd

from ensalada import DATA_PATH, RATS, SESSION_TYPES

MAZE_SCALE_FACTOR = 4.2  # pixels per cm


class Session:

    def __init__(self, rat, session_type='CDEF', day=1, bin_size=0.1):

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

    def position(self):
        def _center_and_scale(a):
            a = a - np.mean([np.nanpercentile(a, p) for p in [5, 95]])
            return a / MAZE_SCALE_FACTOR

        pos = self.position_file.apply(_center_and_scale, axis=0)
        pos['y'] = pos['y'].apply(lambda y: y if np.abs(y) < 50 else np.nan)

        pos.index = pd.to_datetime(pos.index.values, unit='s')

        return pos.resample(
            f"{str(int(1000 * self.bin_size))}ms").apply(np.nanmean)

    def velocity(self, cutoff=25):
        """Calculate from the derivative of the binned position"""
        pos = self.position()
        delta_pos = np.concatenate(
            [[np.nan], np.linalg.norm(np.diff(pos, axis=0), axis=1)])
        delta_pos[delta_pos > cutoff] = np.nan  # remove outrageous numbers

        velocity = pd.Series(delta_pos / self.bin_size, index=pos.index)

        # should probably make smoothing arguments
        return velocity.rolling(window=30, min_periods=20,
                                win_type='gaussian').mean(std=5)

    def detect_interneurons(self, threshold=10):
        """
        Parameters
        ----------
        threshold : float, optional
            Firing rate cutoff for excluding interneurons
        """
        mean_fr = (np.hstack(self.rasters()).mean(axis=1) / self.bin_size)
        return mean_fr > threshold

    def find_iti_periods(self, threshold=20, x_cutoff=50, y_cutoff=10):
        """Find when the animal is in the ITI chamber or back of the contexts,
        and moving relatively slowly"""

        pos = self.position()
        return (
            (np.abs(pos['x']) < x_cutoff)
            & (np.abs(pos['y']) < y_cutoff)
            & (self.velocity() < threshold)
        )

    def labels(self):
        "List of lists, containing string labels for each trial"
        events = self.events_file
        events = events[~events.exclude & events.last_sample & events.correct]

        trial_labels = []
        for _, e in events.iterrows():
            t = []
            t.append('valence_positive' if e.baited else 'valence_negative')
            t.append(f"position_{e.position}")
            t.append(f"context_{e.context}")
            t.append(f"odor_{e.odor}")
            trial_labels.append(t)

        return trial_labels

    def rasters(self, pre=3, post=1.5):
        """
        Parameters
        ----------
        pre, post : float, optional
            Time in seconds to include before sampling onset and after sampling
            offset, respectively

        Returns
        -------
        rasters : list, length (num_trials)
            List of trial rasterplots. Each trial can have a different total
            duration, depending on sampling period length.
        """
        def _list_to_hist(s, start, stop):
            return np.histogram(
                s[(s > start) & (s < stop)],
                bins=np.arange(start, stop + self.bin_size, self.bin_size))[0]

        spikes = self.spikes_file
        events = self.events_file
        events = events[~events.exclude & events.last_sample & events.correct]

        rasters = []
        for _, e in events.iterrows():

            start = e.time - pre
            stop = e.time + e.duration + post

            rasters.append(np.stack(
                spikes.spikes.apply(lambda s: _list_to_hist(s, start, stop)))
            )

        return rasters

    def format_online_data(self, exclude_interneurons=True, **raster_kws):
        """
        Parameters
        ----------
        exclude_interneurons : bool, optional
            Remove high firing rate neurons from the dataset. Defaults to True

        Returns
        -------
        X : array (num_samples, num_units)
            'Flattened' samples matrix, for model fitting.
        y : list of lists, length (num_samples)
            List containing a label list for each sample
        lengths : array, length (num_trials)
            Length of each trial in X, for partitioning
        """

        rasters = self.rasters(**raster_kws)
        trial_labels = self.labels()

        lengths, y = [], []
        for l, r in zip(*[trial_labels, rasters]):
            y += [l] * r.shape[1]
            lengths.append(r.shape[1])

        X = np.hstack(rasters).T

        if exclude_interneurons:
            X = X[:, ~self.detect_interneurons()]

        return X, y, np.array(lengths)

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
