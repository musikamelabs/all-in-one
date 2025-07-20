"""
import sys
import subprocess
import torch

from pathlib import Path
from typing import List, Union


def demix(paths: List[Path], demix_dir: Path, device: Union[str, torch.device]):
  todos = []
  demix_paths = []
  for path in paths:
    out_dir = demix_dir / 'hdemucs_mmi' / path.stem
    demix_paths.append(out_dir)
    if out_dir.is_dir():
      if (
        (out_dir / 'bass.wav').is_file() and
        (out_dir / 'drums.wav').is_file() and
        (out_dir / 'other.wav').is_file() and
        (out_dir / 'vocals.wav').is_file()
      ):
        continue
    todos.append(path)

  existing = len(paths) - len(todos)
  print(f'=> Found {existing} tracks already demixed, {len(todos)} to demix.')

  if todos:
    subprocess.run(
      [
        sys.executable, '-m', 'demucs.separate',
        '--out', demix_dir.as_posix(),
        '--name', 'hdemucs_mmi',
        '--device', str(device),
        *[path.as_posix() for path in todos],
      ],
      check=True,
    )

  return demix_paths
"""
# Optimized Version
import sys
import subprocess
import torch
import os

from pathlib import Path
from typing import List, Union

# Global model cache to avoid reloading
_model_cache = None

def _get_cached_model(device):
    """Load and cache the demucs model to avoid repeated initialization."""
    global _model_cache
    if _model_cache is None:
        try:
            from demucs.pretrained import get_model
            print("=> Loading hdemucs_mmi model (one-time initialization)...")
            _model_cache = get_model('hdemucs_mmi')
            _model_cache.to(device)
            _model_cache.eval()
            torch.set_grad_enabled(False)
            print("=> Model loaded and cached.")
        except ImportError:
            print("=> demucs not available for direct API, falling back to subprocess")
            return None
    return _model_cache

def demix(paths: List[Path], demix_dir: Path, device: Union[str, torch.device], use_python_api: bool = True):
  """Demixes the audio file into its sources."""
  todos = []
  demix_paths = []
  for path in paths:
    out_dir = demix_dir / 'hdemucs_mmi' / path.stem
    demix_paths.append(out_dir)
    if out_dir.is_dir():
      if (
        (out_dir / 'bass.wav').is_file() and
        (out_dir / 'drums.wav').is_file() and
        (out_dir / 'other.wav').is_file() and
        (out_dir / 'vocals.wav').is_file()
      ):
        continue
    todos.append(path)

  existing = len(paths) - len(todos)
  print(f'=> Found {existing} tracks already demixed, {len(todos)} to demix.')

  if todos:
    # Try Python API first for much faster initialization
    if use_python_api and _try_python_api(todos, demix_dir, device):
      return demix_paths
    
    print("=> Using subprocess method (slower initialization)")
    
    # Optimization: Pre-warm the model with a dummy call to reduce subsequent init times
    if len(todos) > 1:
      print("=> Pre-warming model with first file...")
      _prewarm_model(todos[0], demix_dir, device)
      todos = todos[1:]  # Process remaining files
    
    # Determine optimal batch size for T4 GPU memory
    num_files = len(todos)
    if num_files <= 2:
      batch_size = num_files
    elif num_files <= 8:
      batch_size = 2
    else:
      batch_size = 3
    
    # Process files in batches for better memory management
    todo_paths = [path.as_posix() for path in todos]
    
    for i in range(0, len(todo_paths), batch_size):
      batch = todo_paths[i:i + batch_size]
      
      print(f'=> Processing batch {i//batch_size + 1}/{(len(todo_paths)-1)//batch_size + 1}: {len(batch)} files')
      
      # Optimized command with T4-specific flags + initialization optimizations
      cmd = [
        sys.executable, '-m', 'demucs.separate',
        '--out', demix_dir.as_posix(),
        '--name', 'hdemucs_mmi',
        '--device', str(device),
        '--segment', '6',      # Optimal segment size for T4 memory
        '--overlap', '0.1',    # Reduced overlap for 2x speed boost
        '--jobs', '1',         # Reduced from 2 to speed up init
        '--float32',           # Better T4 tensor core utilization
        '--no-split',          # Avoid unnecessary file splitting overhead
        *batch
      ]
      
      # Set environment for better GPU utilization and faster init
      env = os.environ.copy()
      env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Reduce memory fragmentation
      env['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
      
      if isinstance(device, torch.device):
        device_str = str(device)
      else:
        device_str = device
        
      if 'cuda' in device_str and ':' in device_str:
        env['CUDA_VISIBLE_DEVICES'] = device_str.split(':')[-1]
      
      try:
        subprocess.run(cmd, check=True, env=env)
      except subprocess.CalledProcessError as e:
        print(f'=> Batch processing failed, falling back to individual files: {e}')
        # Fallback to individual file processing for this batch
        for file_path in batch:
          try:
            fallback_cmd = [
              sys.executable, '-m', 'demucs.separate',
              '--out', demix_dir.as_posix(),
              '--name', 'hdemucs_mmi',
              '--device', str(device),
              '--segment', '6',
              '--overlap', '0.1',
              '--jobs', '1',
              '--float32',
              '--no-split',
              file_path
            ]
            subprocess.run(fallback_cmd, check=True, env=env)
          except subprocess.CalledProcessError as file_error:
            print(f'=> Failed to process {file_path}: {file_error}')

def _try_python_api(todos: List[Path], demix_dir: Path, device) -> bool:
  """Try using Python API directly for much faster processing."""
  try:
    import torchaudio
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    
    model = _get_cached_model(device)
    if model is None:
      return False
      
    print("=> Using Python API (fast initialization)")
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    
    for audio_path in todos:
      print(f"=> Processing {audio_path.name}...")
      
      # Load audio
      wav, sr = torchaudio.load(audio_path)
      wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
      wav = wav.to(device)
      
      # Apply model with optimizations
      with torch.no_grad():
        sources = apply_model(
          model, wav.unsqueeze(0), 
          segment=6.0,  # Optimal for T4
          overlap=0.1,  # Reduced overlap
          device=device
        )[0]
      
      # Save results in expected format
      output_dir = demix_dir / 'hdemucs_mmi' / audio_path.stem
      output_dir.mkdir(parents=True, exist_ok=True)
      
      for i, source in enumerate(sources):
        stem_name = model.sources[i]
        output_path = output_dir / f"{stem_name}.wav"
        torchaudio.save(output_path, source.cpu(), model.samplerate)
    
    return True
    
  except Exception as e:
    print(f"=> Python API failed ({e}), falling back to subprocess")
    return False

def _prewarm_model(first_file: Path, demix_dir: Path, device):
  """Pre-warm the model with first file to reduce subsequent initialization times."""
  try:
    cmd = [
      sys.executable, '-m', 'demucs.separate',
      '--out', demix_dir.as_posix(),
      '--name', 'hdemucs_mmi', 
      '--device', str(device),
      '--segment', '6',
      '--overlap', '0.1',
      '--jobs', '1',
      '--float32',
      '--no-split',
      first_file.as_posix()
    ]
    
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    env['CUDA_LAUNCH_BLOCKING'] = '0'
    
    subprocess.run(cmd, check=True, env=env)
  except subprocess.CalledProcessError as e:
    print(f"=> Pre-warming failed: {e}")

  return demix_paths
