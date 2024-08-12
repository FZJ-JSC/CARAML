import os
import subprocess
import io
import time
import re
import pandas as pd

rocm_path = os.getenv("ROCM_PATH")
if rocm_path is None:
    rocm_path = "/opt/rocm/"
try:
    from rsmiBindings import *
except:
    import sys
    # 6.x path
    sys.path.append(os.path.join(rocm_path, "libexec/rocm_smi/"))
    # 5.x path
    sys.path.append(os.path.join(rocm_path, "rocm_smi/bindings"))
    from rsmiBindings import *


from multiprocessing import Process, Queue, Event

def power_loop(queue, event, interval):
    init_bindings_required = True
    try:
        with open(os.path.join(rocm_path,'.info/version'), 'r') as vfile:
            vstr = vfile.readline()
            print(f"ROCM version: {vstr}")
            vmaj = int(re.search(r'\d+', vstr).group())
            if vmaj < 6:
                init_bindings_required = False
    except:
        init_bindings_required = False
    if init_bindings_required:
       rocmsmi_init = initRsmiBindings(silent=False)
    else:
        rocmsmi_init = rocmsmi
    ret = rocmsmi_init.rsmi_init(0)
    if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
        raise RuntimeError("Failed initializing rocm_smi library")
    device_count = c_uint32(0)
    ret = rocmsmi_init.rsmi_num_monitor_devices(byref(device_count))
    if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
        raise RuntimeError("Failed enumerating ROCm devices")
    device_list = list(range(device_count.value))
    power_value_dict = {
        id : [] for id in device_list
    }
    power_value_dict['timestamps'] = []
    last_timestamp = time.time()
    dev_pwr_map = { id: True for id in device_list }
    start_energy_list = []
    for id in device_list:
        energy = c_uint64()
        energy_timestamp = c_uint64()
        energy_resolution = c_float()
        ret = rocmsmi_init.rsmi_dev_energy_count_get(id, 
                byref(energy),
                byref(energy_resolution),
                byref(energy_timestamp))
        if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
            raise RuntimeError(f"Failed getting Power of device {id}")
        if 0 == energy.value:
            dev_pwr_map[id] = False
        start_energy_list.append(round(energy.value*energy_resolution.value,2)) # unit is uJ

    while not event.is_set():
        for id in device_list:
            power = c_uint32()
            if not dev_pwr_map[id]:
                power.value = 0
            else:
                ret = rocmsmi_init.rsmi_dev_power_ave_get(id, 0, byref(power))
                if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
                    raise RuntimeError(f"Failed getting power of device {id}: {ret}")
            power_value_dict[id].append(power.value*1e-6) # value is uW
        timestamp = time.time()
        power_value_dict['timestamps'].append(timestamp)
        wait_for = max(0,1e-3*interval-(timestamp-last_timestamp))
        time.sleep(wait_for)
        last_timestamp = timestamp

    energy_list = [0.0 for _ in device_list]
    for id in device_list:
        energy = c_uint64()
        energy_timestamp = c_uint64()
        energy_resolution = c_float()
        ret = rocmsmi_init.rsmi_dev_energy_count_get(id, 
                byref(energy),
                byref(energy_resolution),
                byref(energy_timestamp))
        if rsmi_status_t.RSMI_STATUS_SUCCESS != ret:
            raise RuntimeError(f"Failed getting Power of device {id}")
        energy_list[id] = round(energy.value*energy_resolution.value,2) - start_energy_list[id]

    energy_list = [ (energy*1e-6)/3600 for energy in energy_list] # convert uJ to Wh
    queue.put(power_value_dict)
    queue.put(energy_list)

class GetPower(object):
    def __enter__(self):
        self.end_event = Event()
        self.power_queue = Queue()
        
        interval = 100 #ms
        self.smip = Process(target=power_loop,
                args=(self.power_queue, self.end_event, interval))
        self.smip.start()
        return self
    def __exit__(self, type, value, traceback):
        self.end_event.set()
        power_value_dict = self.power_queue.get()
        self.energy_list_counter = self.power_queue.get()
        self.smip.join()

        self.df = pd.DataFrame(power_value_dict)
    def energy(self):
        import numpy as np
        _energy = []
        energy_df = self.df.loc[:,self.df.columns != 'timestamps'].astype(float).multiply(self.df["timestamps"].diff(),axis="index")/3600
        _energy = energy_df[1:].sum(axis=0).values.tolist()
        return _energy,self.energy_list_counter


if __name__ == "__main__":
    with GetPower() as measured_scope:
        print('Measuring Energy during main() call')
        try:
            main(args)
        except Exception as exc:
            import traceback
            print(f"Errors occured during training: {exc}")
            print(f"Traceback: {traceback.format_exc()}")
    energy_int,energy_cnt = measured_scope.energy()
    print(f"Energy-per-GPU-list integrated(Wh): {energy_int}")
    print(f"Energy-per-GPU-list from counter(Wh): {energy_cnt}")
