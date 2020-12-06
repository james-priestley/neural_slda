import os
from functools import lru_cache

import pandas as pd

from ensalada import DATA_PATH, RATS, SESSION_TYPES


class Session:
    
    def __init__(self, rat, session_type='CDEF', day=1):
        
        assert rat in RATS, "Invalid rat name!"
        assert session_type in SESSION_TYPES, "Invalid session type!"
        
        self.rat = rat
        self.session_type = session_type
        self.day = day
        self.data_path = os.path.join(DATA_PATH, rat, session_type + str(day))
    
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
    def num_units(self):
        return len(self.spikes_file)
    
    @property
    def num_trials(self):
        pass

class Rat(list):
    
    def __init__(self, rat):
        
        assert rat in RATS, "Invalid rat name!"
        
        # init a session object for all sessions and init list