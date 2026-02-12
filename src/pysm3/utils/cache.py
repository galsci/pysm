import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from .. import units as u

log = logging.getLogger("pysm3")


def get_cache_filename(preset_string, nside, freq_ghz):
    """Generate a canonical cache filename for a preset/frequency/nside combination."""
    return f"{preset_string}_{nside}_{freq_ghz:.1f}GHz.npz"


@u.quantity_input
def get_or_create_cached_preset_map(
    preset_string,
    nside,
    freq: u.Quantity[u.GHz],
    cache_dir,
    output_unit=u.uK_RJ,
    dtype=np.float32,
):
    """Load a cached preset map or generate it with ``Sky`` and cache it.

    Parameters
    ----------
    preset_string : str
        PySM preset tag, for example ``d10`` or ``s5``.
    nside : int
        Output NSIDE used to generate or load the cached map.
    freq : Quantity
        Frequency in units compatible with GHz.
    cache_dir : str or Path
        Folder where the compressed ``.npz`` cache files are stored.
    output_unit : Unit
        Unit for the cached map, defaults to ``uK_RJ``.
    dtype : numpy dtype
        Data type used when writing cache files, defaults to ``float32``.

    Returns
    -------
    numpy.ndarray
        Cached or freshly generated IQU map with shape ``(3, npix)``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    freq_ghz = u.Quantity(freq).to_value(u.GHz)
    cache_path = cache_dir / get_cache_filename(preset_string, nside, freq_ghz)

    if cache_path.exists():
        return np.load(cache_path)["map"]

    from ..sky import Sky

    sky = Sky(nside=nside, preset_strings=[preset_string], output_unit=output_unit)
    output_map = sky.get_emission(freq).to_value(output_unit).astype(dtype)
    np.savez_compressed(cache_path, map=output_map)
    return output_map


@u.quantity_input
def get_or_create_cached_preset_map_subprocess(
    preset_string,
    nside,
    freq: u.Quantity[u.GHz],
    cache_dir,
    output_unit=u.uK_RJ,
    dtype=np.float32,
):
    """Build/load preset map cache in a subprocess and return cached data.

    This avoids large template initialization in the caller process (e.g. a
    Jupyter kernel), reducing risk of kernel crashes due to transient memory
    spikes while loading heavy models. If the subprocess exits unexpectedly
    (for example killed by the OS), this function falls back to in-process
    generation.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    freq_ghz = u.Quantity(freq).to_value(u.GHz)
    cache_path = cache_dir / get_cache_filename(preset_string, nside, freq_ghz)

    if not cache_path.exists():
        payload = json.dumps(
            {
                "preset_string": preset_string,
                "nside": int(nside),
                "freq_ghz": float(freq_ghz),
                "cache_dir": str(cache_dir),
                "output_unit": str(output_unit),
                "dtype": np.dtype(dtype).name,
            }
        )
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pysm3.utils.cache import _cache_cli; _cache_cli()",
                    payload,
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            log.warning(
                "Subprocess cache generation failed for %s at %.1f GHz, retrying in-process",
                preset_string,
                freq_ghz,
            )
            get_or_create_cached_preset_map(
                preset_string=preset_string,
                nside=nside,
                freq=freq,
                cache_dir=cache_dir,
                output_unit=output_unit,
                dtype=dtype,
            )

    return np.load(cache_path)["map"]


def _cache_cli():
    payload = json.loads(sys.argv[1])
    get_or_create_cached_preset_map(
        preset_string=payload["preset_string"],
        nside=payload["nside"],
        freq=float(payload["freq_ghz"]) * u.GHz,
        cache_dir=payload["cache_dir"],
        output_unit=u.Unit(payload["output_unit"]),
        dtype=np.dtype(payload["dtype"]),
    )
