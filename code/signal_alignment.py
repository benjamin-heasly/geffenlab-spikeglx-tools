import numpy as np
import os
from pathlib import Path

# Description:
'''Batch process spike time mapping from OneBox to IMEC timebase for a recording session. Extracts treadmill data.'''

def convert_spike_times_to_seconds(spike_times_path, metadata_path):
    """Convert spike times from samples to seconds."""
    spike_times = np.load(spike_times_path)
    
    with open(metadata_path, 'r') as file:
        metadata = file.readlines()
    
    imSampRate = None
    for line in metadata:
        if line.startswith('imSampRate'):
            imSampRate = float(line.split('=')[1].strip())
            break
    
    if imSampRate is None:
        raise Valueerror("Sample rate (imSampRate) not found in metadata file.")
    
    spike_times_seconds = spike_times / imSampRate
    output_path = os.path.join(os.path.dirname(spike_times_path), 'spike_seconds.npy')
    np.save(output_path, spike_times_seconds)
    
    print(f"Converted spike times saved to {output_path}")
    return spike_times_seconds

def map_events(fromstream_edges, tostream_edges, event_times):
    """Map event times from OneBox to IMEC timebase."""
    mapped_events = np.zeros_like(event_times)
    
    for i, event_time in enumerate(event_times):
        idx = np.searchsorted(fromstream_edges, event_time, side='right') - 1
        
        if idx < 0 or idx >= len(tostream_edges):
            mapped_events[i] = np.nan
        else:
            A_edge = fromstream_edges[idx]
            B_edge = tostream_edges[idx]
            mapped_events[i] = event_time - A_edge + B_edge
    
    return mapped_events

def load_onebox_continuous_channel(obx_binary_path, channel_idx, metadata_path, num_channels=14):
    """
    Extract continuous analog data from OneBox binary file.
    
    Args:
        obx_binary_path (str): Path to .obx.bin file
        channel_idx (int): Channel to extract (0-13)
        metadata_path (str): Path to .obx.meta file
        num_channels (int): Total channels in file
    
    Returns:
        tuple: (voltage_data, sample_rate)
    """
    
    # Read metadata
    with open(metadata_path, 'r') as f:
        metadata = f.readlines()
    
    sample_rate = None
    ai_range_max = None
    ai_range_min = None
    max_int = None
    
    for line in metadata:
        if line.startswith('obSampRate='):
            sample_rate = float(line.split('=')[1].strip())
        elif line.startswith('obAiRangeMax='):
            ai_range_max = float(line.split('=')[1].strip())
        elif line.startswith('obAiRangeMin='):
            ai_range_min = float(line.split('=')[1].strip())
        elif line.startswith('obMaxInt='):
            max_int = float(line.split('=')[1].strip())
    
    # Load binary data
    data = np.fromfile(obx_binary_path, dtype=np.int16)
    data = data.reshape(-1, num_channels)
    
    # Extract channel
    channel_data = data[:, channel_idx].astype(np.float32)
    
    # Convert to voltage
    voltage_range = ai_range_max - ai_range_min
    voltage_data = (channel_data / max_int) * voltage_range + ai_range_min
    
    print(f"Loaded channel {channel_idx}: {len(voltage_data)} samples at {sample_rate} Hz")
    
    return voltage_data, sample_rate

def process_session(base_dir, imec_probe='imec0'):
    """
    Process a single recording session.
    
    Args:
        base_dir (str): Base directory (e.g., "D:\AD020\AD020_260128_g0")
        imec_probe (str): IMEC probe name (default: 'imec0')
    """
    base_dir = Path(base_dir)
    print(f"\n{'='*60}")
    print(f"Processing: {base_dir.name}")
    print(f"{'='*60}")
    
    # Construct paths
    session_name = base_dir.name
    imec_dir = base_dir / f"{session_name}_{imec_probe}"
    kilosort_dir = imec_dir / "kilosort4"
    catgt_dir = base_dir / f"catgt_{session_name}"
    
    # Check if required directories exist
    if not kilosort_dir.exists():
        print(f"error: kilosort4 directory not found at {kilosort_dir}")
        return False
    
    if not catgt_dir.exists():
        print(f"error: catgt directory not found at {catgt_dir}")
        return False
    
    # Step 1: Convert spike times to seconds
    print("\n Converting spike times to seconds...")
    spike_times_path = kilosort_dir / "spike_times.npy"
    metadata_path = imec_dir / f"{session_name}_t0.{imec_probe}.ap.meta"
    
    if not spike_times_path.exists():
        print(f"error: spike_times.npy not found at {spike_times_path}")
        return False
    
    if not metadata_path.exists():
        print(f"error: metadata file not found at {metadata_path}")
        return False
    
    spike_times_seconds = convert_spike_times_to_seconds(str(spike_times_path), str(metadata_path))
    
    # Step 2: Load sync data
    print("\n Loading synchronization data...")
    fromstream_edges_path = catgt_dir / f"{session_name}_tcat.obx0.obx.xd_13_6_500.txt"
    tostream_edges_path = catgt_dir / f"{session_name}_imec0" / f"{session_name}_tcat.{imec_probe}.ap.xd_384_6_500.txt"
    event_times_path = catgt_dir / f"{session_name}_tcat.obx0.obx.xa_0_0.txt"
    
    for path in [fromstream_edges_path, tostream_edges_path, event_times_path]:
        if not path.exists():
            print(f"error: Required file not found at {path}")
            return False
    
    fromstream_edges = np.loadtxt(fromstream_edges_path)
    tostream_edges = np.loadtxt(tostream_edges_path)
    event_times = np.loadtxt(event_times_path)
    
    # Step 3: Map events
    print("\n Mapping events and spikes...")
    mapped_event_times = map_events(fromstream_edges, tostream_edges, event_times)
    mapped_spike_times = map_events(fromstream_edges, tostream_edges, spike_times_seconds)
    
    # Step 4: Load and map treadmill data
    print("\n Loading and mapping treadmill data...")
    obx_binary_path = base_dir / f"{session_name}_t0.obx0.obx.bin"
    obx_metadata_path = base_dir / f"{session_name}_t0.obx0.obx.meta"
    
    if not obx_binary_path.exists():
        print(f"error: OneBox binary file not found at {obx_binary_path}")
        return False
    
    if not obx_metadata_path.exists():
        print(f"error: OneBox metadata file not found at {obx_metadata_path}")
        return False
    
    treadmill_voltage, treadmill_sample_rate = load_onebox_continuous_channel(
        str(obx_binary_path), 
        channel_idx=7, 
        metadata_path=str(obx_metadata_path)
    )
    
    # Create time array for treadmill
    treadmill_times = np.arange(len(treadmill_voltage)) / treadmill_sample_rate
    
    # Map treadmill times to IMEC timebase
    mapped_treadmill_times = map_events(fromstream_edges, tostream_edges, treadmill_times)
    
    print(f"Treadmill voltage range: {treadmill_voltage.min():.2f}V to {treadmill_voltage.max():.2f}V")

    # Step 5: Save results
    print("\n[Step 5] Saving results...")
    output_dir = kilosort_dir
    np.savetxt(output_dir / 'mapped_event_times.txt', mapped_event_times, fmt='%.6f', delimiter='\t')
    np.save(output_dir / 'mapped_spike_times.npy', mapped_spike_times)
    np.save(output_dir / 'treadmill_voltage.npy', treadmill_voltage)
    np.savetxt(output_dir / 'mapped_treadmill_times.txt', mapped_treadmill_times, fmt='%.6f', delimiter='\t')
    
    print(f"Results saved to {output_dir}")
    print(f"mapped_event_times.txt")
    print(f"mapped_spike_times.npy")
    print(f"treadmill_voltage.npy")
    print(f"mapped_treadmill_times.txt")
    
    return True
    
if __name__ == "__main__":
    # Configuration
    BASE_DIR = "D:\AD021\AD021_260219_g0"
    IMEC_PROBE = "imec0"
    
    # Run processing
    success = process_session(BASE_DIR, IMEC_PROBE)
    
    if success:
        print("\n Spike time mapping completed successfully")
    else:
        print("\n Unable to find path or other error occurred")
