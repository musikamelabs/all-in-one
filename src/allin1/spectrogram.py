"""
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from multiprocessing import Pool
from madmom.audio.signal import FramedSignalProcessor, Signal
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor


def extract_spectrograms(demix_paths: List[Path], spec_dir: Path, multiprocess: bool = True):
  todos = []
  spec_paths = []
  for src in demix_paths:
    dst = spec_dir / f'{src.stem}.npy'
    spec_paths.append(dst)
    if dst.is_file():
      continue
    todos.append((src, dst))

  existing = len(spec_paths) - len(todos)
  print(f'=> Found {existing} spectrograms already extracted, {len(todos)} to extract.')

  if todos:
    # Define a pre-processing chain, which is copied from madmom.
    frames = FramedSignalProcessor(
      frame_size=2048,
      fps=int(44100 / 441)
    )
    stft = ShortTimeFourierTransformProcessor()  # caching FFT window
    filt = FilteredSpectrogramProcessor(
      num_bands=12,
      fmin=30,
      fmax=17000,
      norm_filters=True
    )
    spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
    processor = SequentialProcessor([frames, stft, filt, spec])

    # Process all tracks using multiprocessing.
    if multiprocess:
      pool = Pool()
      map_fn = pool.imap
    else:
      pool = None
      map_fn = map

    iterator = map_fn(_extract_spectrogram, [
      (src, dst, processor)
      for src, dst in todos
    ])
    for _ in tqdm(iterator, total=len(todos), desc='Extracting spectrograms'):
      pass

    if pool:
      pool.close()
      pool.join()

  return spec_paths


def _extract_spectrogram(args: Tuple[Path, Path, SequentialProcessor]):
  src, dst, processor = args

  dst.parent.mkdir(parents=True, exist_ok=True)

  sig_bass = Signal(src / 'bass.wav', num_channels=1)
  sig_drums = Signal(src / 'drums.wav', num_channels=1)
  sig_other = Signal(src / 'other.wav', num_channels=1)
  sig_vocals = Signal(src / 'vocals.wav', num_channels=1)

  spec_bass = processor(sig_bass)
  spec_drums = processor(sig_drums)
  spec_others = processor(sig_other)
  spec_vocals = processor(sig_vocals)

  spec = np.stack([spec_bass, spec_drums, spec_others, spec_vocals])  # instruments, frames, bins

  np.save(str(dst), spec)
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from ray import tune
from madmom.audio.signal import FramedSignalProcessor, Signal
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor

import scipy.fftpack as scipy_fft


def extract_spectrograms(demix_paths: List[Path], spec_dir: Path, multiprocess: bool = True):
    todos = []
    spec_paths = []
    for src in demix_paths:
        dst = spec_dir / f'{src.stem}.npy'
        spec_paths.append(dst)
        if dst.is_file():
            continue
        todos.append((src, dst))

    existing = len(spec_paths) - len(todos)
    print(f"=> Found {existing} spectrograms already extracted, {len(todos)} to extract.")

    if todos:
        # Define a pre-processing chain, which is copied from madmom.
        frames = FramedSignalProcessor(
            frame_size=2048,
            fps=int(44100 / 441)
        )

        # Pre-compute the FFT window for efficient processing.
        fft_window = scipy_fft.fftshift(scipy_fft.hamming(2048))

        stft = ShortTimeFourierTransformProcessor(window=fft_window)
        filt = FilteredSpectrogramProcessor(
            num_bands=12,
            fmin=30,
            fmax=17000,
            norm_filters=True
        )
        spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
        processor = SequentialProcessor([frames, stft, filt, spec])

        # Utilize Ray for efficient multiprocessing.
        if multiprocess:
            tune.run(
                _extract_spectrogram_ray,
                num_cpus=1,
                resources_per_trial={"cpu": 1},
                config={"demix_path": src, "dst_path": dst, "processor": processor}
            )
        else:
            # Use a more memory-efficient data structure.
            spectrograms = []

            for src, dst in todos:
                spec_bass, spec_drums, spec_others, spec_vocals = processor(Signal(src / 'bass.wav', num_channels=1))
                spectrograms.append((src.stem, (spec_bass, spec_drums, spec_others, spec_vocals)))

            np.save(spec_dir / 'spectrograms.npy', spectrograms)

    # Return the paths to extracted spectrograms
    return spec_paths


@ray.remote(num_cpus=1)
def _extract_spectrogram_ray(demix_path: Path, dst_path: Path, processor: SequentialProcessor):
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Process the audio track and save the spectrogram.
    sig_bass = Signal(demix_path / 'bass.wav', num_channels=1)
    sig_drums = Signal(demix_path / 'drums.wav', num_channels=1)
    sig_others = Signal(demix_path / 'other.wav', num_channels=1)
    sig_vocals = Signal(demix_path / 'vocals.wav', num_channels=1)

    spec_bass = processor(sig_bass)
    spec_drums = processor(sig_drums)
    spec_others = processor(sig_others)
    spec_vocals = processor(sig_vocals)

    spectrogram = np.stack([spec_bass, spec_drums, spec_others, spec_vocals])

    np.save(str(dst_path), spectrogram)
