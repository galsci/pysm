import psutil
import gc
import numpy as np
import time

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
        memstr = "***** {}\n".format(msg)
        memstr += "Walltime elapsed since the beginning {:.1f} s\n".format(time.time() - self.timer)
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
                    memstr += "{:>12} : {:8.3f} {}\n".format(key, vlist[0], unit)
                    if np.abs(vlist2[0] - vlist[0]) / vlist[0] > 1e-3:
                        memstr += "{:>12} : {:8.3f} {} (after GC)\n".format(
                            key, vlist2[0], unit
                        )
                else:
                    med1 = np.median(vlist)
                    memstr += (
                        "{:>12} : {:8.3f} {}  < {:8.3f} +- {:8.3f} {}  "
                        "< {:8.3f} {}\n".format(
                            key,
                            np.amin(vlist),
                            unit,
                            med1,
                            np.std(vlist),
                            unit,
                            np.amax(vlist),
                            unit,
                        )
                    )
                    med2 = np.median(vlist2)
                    if np.abs(med2 - med1) / med1 > 1e-3:
                        memstr += (
                            "{:>12} : {:8.3f} {}  < {:8.3f} +- {:8.3f} {}  "
                            "< {:8.3f} {} (after GC)\n".format(
                                key,
                                np.amin(vlist2),
                                unit,
                                med2,
                                np.std(vlist2),
                                unit,
                                np.amax(vlist2),
                                unit,
                            )
                        )
        if self.comm is None or self.comm.rank == 0:
            print(memstr, flush=True)
        if self.comm is not None:
            self.comm.Barrier()
