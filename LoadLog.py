import os
import warnings
from pathlib import Path
from collections import UserList
from tensorboard.backend.event_processing import event_accumulator  
from tensorboard.plugins.hparams import plugin_data_pb2

class SingleLog():
    '''
    mode: set to 'single_scalar' if the log contains scalar events only
          set to  'scalar+hparams' if the log contains both scalar events and hparams events. 
    scalar_ea: event_accumulator for scalar
    hparams_ea: event_accumulator for hparams and metrics
    '''
    def __init__(self, path):
        path = Path(path)
        self.name = path.name
        if path.is_file():
            self.mode = 'single_scalar'
            self.scalar_path = path
        else:
            sub_files = [sub for sub in path.iterdir() if sub.is_file()]
            sub_dirs = [sub for sub in path.iterdir() if sub.is_dir()]
            assert len(sub_files) >=  1, f"no log in Single Log {self.name}"
            assert len(sub_files) == len(sub_dirs) or len(sub_dirs) == 0, f"scalar log number cannot match hparam log number! In single log {self.name}"
            if len(sub_files) > 1:
                warnings.warn('More than one logs in the single log, the latest one choosed.')
            self.scalar_path = sorted(sub_files)[-1] # choose the Latest 
            if len(sub_dirs) == 0:
                self.mode = 'single_scalar'
            else:
                self.mode = 'scalar+hparams'
                self.hparams_path = list(sorted(sub_dirs)[-1].iterdir())[0] # choose the Latest 
        
        self.scalar_ea = event_accumulator.EventAccumulator(str(self.scalar_path))
        self.scalar_ea.Reload()
        
        if self.mode == 'scalar+hparams':
            self.hparams_ea = event_accumulator.EventAccumulator(str(self.hparams_path))
            self.hparams_ea.Reload()

    def keys(self):
        '''returns the keys for the scalar events in the log file/directory.
        '''
        return self.scalar_ea.scalars.Keys()
    
    def get_scalar(self, key:str, default=None, raw_ScalarEvent=False):
        '''takes in a scalar key and returns the value(s) associated with that key. 
        raw_ScalarEvent: return raw ScalarEvent objects if stored True
        '''
        if key not in self.keys():
            return default
        else:
            scalarEvent_logs = self.scalar_ea.scalars.Items(key)
            return scalarEvent_logs if raw_ScalarEvent else [l.value for l in scalarEvent_logs]
    
    def m_keys(self):
        '''returns the keys for the hparams in the log file/directory.
        '''
        return self.hparams_ea.scalars.Keys()
    
    def get_metric(self, m_key:str, default=None, raw_ScalarEvent=False):
        '''takes in a metric key and returns the value(s) associated with that key. 
        raw_ScalarEvent: return raw ScalarEvent objects if stored True
        '''
        assert self.mode == 'scalar+hparams'
        if m_key not in self.m_keys():
            return default
        else:
            scalarEvent_logs = self.hparams_ea.scalars.Items(m_key)
            return scalarEvent_logs if raw_ScalarEvent else [l.value for l in scalarEvent_logs]

    @property
    def hparams(self):
        '''returns a dictionary of the hyperparameters used in the training.'''
        assert self.mode == 'scalar+hparams'
        bstr = self.hparams_ea.PluginTagToContent("hparams")['_hparams_/session_start_info']
        pluginData_result = plugin_data_pb2.HParamsPluginData.FromString(bstr)
        messageMapContainer_result = pluginData_result.session_start_info.hparams
        _result = dict(messageMapContainer_result)
        return {k:v.ListFields()[-1][-1] for k,v in _result.items()}

class GroupLog(UserList[SingleLog]):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.name = self.path.name
        assert self.path.is_dir(), f"{self.path} is not a directory"
        sub_dirs = [sub for sub in self.path.iterdir() if sub.is_dir()]
        for dir_path in sub_dirs:
            self.append(SingleLog(dir_path))
        
    @property
    def sub_names(self):
        return [log.name for log in self.data]
    def keys(self):
        return self.data[0].keys()
    def get_scalar(self, key):
        return [log.get_scalar(key) for log in self.data]
    def m_keys(self):
        return self.data[0].m_keys()
    def get_metric(self, m_key):
        return [log.get_metric(m_key) for log in self.data]


