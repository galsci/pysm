import gc
import time

import numpy as np
import psutil


class MemReporter:

    def __init__(self, comm=None):
        self.comm = comm
        self.timer = time.time()

    def run(self, msg=""):
        """ Gather and report the amount of allocated, free and swapped system memory
        """
        vmem = psutil.virtual_memory()._asdict()
        gc.collect()
        vmem2 = psutil.virtual_memory()._asdict()
        memstr = f"***** {msg}\n"
        memstr += f"Walltime elapsed since the beginning {time.time() - self.timer:.1f} s\n"
        memstr += "Memory usage \n"
        for key, value in vmem.items():
            value2 = vmem2[key]
            if self.comm is None:
                vlist = [value]
                vlist2 = [value2]
            else:
                vlist = self.comm.gather(value)
                vlist2 = self.comm.gather(value2)
            if self.comm is None or self.comm.rank == 0:
                vlist = np.array(vlist, dtype=np.float64)
                vlist2 = np.array(vlist2, dtype=np.float64)
                if key != "percent":
                    # From bytes to better units
                    if np.amax(vlist) < 2 ** 20:
                        vlist /= 2 ** 10
                        vlist2 /= 2 ** 10
                        unit = "kB"
                    elif np.amax(vlist) < 2 ** 30:
                        vlist /= 2 ** 20
                        vlist2 /= 2 ** 20
                        unit = "MB"
                    else:
                        vlist /= 2 ** 30
                        vlist2 /= 2 ** 30
                        unit = "GB"
                else:
                    unit = "% "
                if self.comm is None or self.comm.size == 1:
                    memstr += f"{key:>12} : {vlist[0]:8.3f} {unit}\n"
                    if np.abs(vlist2[0] - vlist[0]) / vlist[0] > 1e-3:
                        memstr += f"{key:>12} : {vlist2[0]:8.3f} {unit} (after GC)\n"
                else:
                    med1 = np.median(vlist)
                    memstr += (
                        f"{key:>12} : {np.amin(vlist):8.3f} {unit}  < {med1:8.3f} +- {np.std(vlist):8.3f} {unit}  "
                        f"< {np.amax(vlist):8.3f} {unit}\n"
                    )
                    med2 = np.median(vlist2)
                    if np.abs(med2 - med1) / med1 > 1e-3:
                        memstr += (
                            f"{key:>12} : {np.amin(vlist2):8.3f} {unit}  < {med2:8.3f} +- {np.std(vlist2):8.3f} {unit}  "
                            f"< {np.amax(vlist2):8.3f} {unit} (after GC)\n"
                        )
        if self.comm is None or self.comm.rank == 0:
            print(memstr, flush=True)
        if self.comm is not None:
            self.comm.Barrier()
