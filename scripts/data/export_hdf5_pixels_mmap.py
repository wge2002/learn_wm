"""Export an HDF5 pixels column as a lossless NumPy mmap sidecar.

Large compressed HDF5 chunks are efficient for sequential storage but costly
for random clip sampling: reading a few frames can decompress an entire chunk.
The generated ``.npy`` file stores the exact uint8 pixels contiguously so
``HDF5Dataset(pixels_path=...)`` can fetch only the requested frames while the
remaining columns continue to come from the original HDF5 file.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np


def export_pixels(
    source: Path,
    output: Path,
    *,
    batch_frames: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'{output.name}.tmp')

    with h5py.File(source, 'r', swmr=True) as h5:
        pixels = h5['pixels']
        shape = pixels.shape
        dtype = pixels.dtype

        if output.exists():
            existing = np.load(output, mmap_mode='r')
            matches = existing.shape == shape and existing.dtype == dtype
            del existing
            if matches:
                print(f'Already complete: {output}')
                return
            raise ValueError(
                f'Existing output {output} does not match '
                f'shape={shape}, dtype={dtype}'
            )

        if temporary.exists():
            temporary.unlink()

        mmap = np.lib.format.open_memmap(
            temporary,
            mode='w+',
            dtype=dtype,
            shape=shape,
        )
        started = time.perf_counter()
        total = shape[0]
        for start in range(0, total, batch_frames):
            end = min(start + batch_frames, total)
            mmap[start:end] = pixels[start:end]
            if end == total or start % (10 * batch_frames) == 0:
                elapsed = time.perf_counter() - started
                rate = end / max(elapsed, 1e-9)
                remaining = (total - end) / max(rate, 1e-9)
                print(
                    f'{end}/{total} frames '
                    f'({100 * end / total:.1f}%), '
                    f'{rate:.1f} frame/s, ETA {remaining / 60:.1f} min',
                    flush=True,
                )

        mmap.flush()
        del mmap

    os.replace(temporary, output)
    print(f'Exported lossless pixels sidecar: {output}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--batch-frames', type=int, default=1000)
    args = parser.parse_args()
    if args.batch_frames <= 0:
        parser.error('--batch-frames must be positive')
    export_pixels(
        args.source.expanduser().resolve(),
        args.output.expanduser().resolve(),
        batch_frames=args.batch_frames,
    )


if __name__ == '__main__':
    main()
