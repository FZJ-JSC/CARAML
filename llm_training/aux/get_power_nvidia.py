import time
import io, os, subprocess, signal
import pandas as pd
from multiprocessing import Process, Queue, Event
import sys

class GetPower(object):
    def __enter__(self):
        cmd = "nvidia-smi --query-gpu=index,timestamp,name,power.draw --format=csv --loop-ms=50"
        self.smi = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True, preexec_fn=os.setsid)
        return self
    def __exit__(self, type, value, traceback):
        #self.smi.kill()
        os.killpg(os.getpgid(self.smi.pid), signal.SIGTERM) #https://stackoverflow.com/a/4791612
        outs, errs = self.smi.communicate()
        self.data = io.StringIO(outs)
        self.df = pd.read_csv(self.data, parse_dates=[1], skipinitialspace=True)
    def energy(self):
        import numpy as np
        _energy = []
        for df_name, df_group in self.df.groupby('index'):
            _energy.append(np.sum(df_group['power.draw [W]'].str.replace(' W', '').astype(float) * df_group['timestamp'].diff().dt.total_seconds())/3600)
        return _energy

class GetNVIDIAPower(object):
    def __enter__(self):
        self.end_event = Event()
        self.power_queue = Queue()

        interval = 100 #ms
        self.smip = Process(target=self._power_loop,
                args=(self.power_queue, self.end_event, interval))
        self.smip.start()
        return self

    def _power_loop(self,queue, event, interval):
        import pynvml as pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        device_list = [pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in range(device_count)]
        power_value_dict = {
                idx : [] for idx in range(device_count)
                }
        power_value_dict['timestamps'] = []
        last_timestamp = time.time()

        while not event.is_set():
            for idx,handle in enumerate(device_list):
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_value_dict[idx].append(power*1e-3)
            timestamp = time.time()
            power_value_dict['timestamps'].append(timestamp)
            wait_for = max(0,1e-3*interval-(timestamp-last_timestamp))
            time.sleep(wait_for)
            last_timestamp = timestamp
        queue.put(power_value_dict)

    def __exit__(self, type, value, traceback):
        self.end_event.set()
        power_value_dict = self.power_queue.get()
        self.smip.join()

        self.df = pd.DataFrame(power_value_dict)

    def energy(self):
        import numpy as np
        _energy = []
        energy_df = self.df.loc[:,self.df.columns != 'timestamps'].astype(float).multiply(self.df["timestamps"].diff(),axis="index")/3600
        _energy = energy_df[1:].sum(axis=0).values.tolist()
        return _energy
