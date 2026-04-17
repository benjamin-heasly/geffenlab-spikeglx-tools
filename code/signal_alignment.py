import sys
from pathlib import Path
import logging
from argparse import ArgumentParser
from typing import Optional, Sequence

import numpy as np

from files import find_one


def set_up_logging():
    logging.root.handlers = []
    handlers = [
        logging.StreamHandler(sys.stdout)
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )


def parse_meta(
    meta_path: Path
) -> dict[str, int | float | str]:
    """Parse a SpikeGLX .meta file into a Python dict."""

    meta_info = {}
    with open(meta_path, 'r') as f:
        for line in f:
            line_parts = line.split("=", maxsplit=1)
            key = line_parts[0].strip()
            if len(line_parts) > 1:
                raw_value = line_parts[1].strip()
                try:
                    value = int(raw_value)
                except:
                    try:
                        value = float(raw_value)
                    except:
                        value = raw_value
            else:
                value = None
            meta_info[key] = value
    return meta_info


def map_events(
    from_events: np.ndarray,
    from_sync: np.ndarray,
    to_sync: np.ndarray,
) -> np.ndarray:
    """Map the event times in from_events from one time stream to another, based on corresponding sync event times in from_sync and to_sync."""

    # For each from event, what was the nearest preceeding sync event on the same stream?
    sync_indices = np.searchsorted(from_sync, from_events, side='right') - 1

    # Any from events that came before the first sync event, use the first sync event.
    sync_indices[sync_indices < 0] = 0

    # Correct each from event based on pairs of corresponding sync events.
    stream_corrections = to_sync[sync_indices] - from_sync[sync_indices]
    to_events = from_events + stream_corrections

    return to_events


def load_voltages(
    bin_path: Path,
    channel_count: int,
    channel_index: int,
    raw_max: float,
    voltage_max: float,
    voltage_min: float,
    raw_dtype=np.int16,
    voltage_dtype=np.int32
) -> np.ndarray:
    """Load samples data from a raw .bin file, scale to voltages."""

    # Load the raw data as one big array with shape (sample_count, channel_count).
    # If this proves cumbersome, we could memmap a chunk at a time and write it out here.
    raw_samples = np.fromfile(bin_path, dtype=raw_dtype)
    raw_samples = raw_samples.reshape(-1, channel_count)

    # Select one channel to convert to voltages.
    raw_channel = raw_samples[:, channel_index].astype(voltage_dtype)
    voltage_range = voltage_max - voltage_min
    voltage_data = (raw_channel / raw_max) * voltage_range + voltage_min
    return voltage_data


def align_signal(
    data_path: Path,
    catgt_path: Path,
    output_path: Path,
    meta_pattern: str,
    bin_pattern: str,
    from_sync_pattern: str,
    to_sync_pattern: str,
    channel_count_meta_name: str,
    voltage_max_meta_name: str,
    voltage_min_meta_name: str,
    raw_max_meta_name: str,
    sample_rate_meta_name: str,
    channel_index: int,
    channel_name: str,
):
    """Locate SpikeGlx/CatGT runs, use sunc pulse times to align samples from a continuous signal recording."""

    # Locate SpikeGlx runs as subfolders of the CATGT_ROOT.
    logging.info(f"Processing SpikeGlx/CatGT runs within: {catgt_path}")
    catgt_run_paths = [run_dir for run_dir in catgt_path.iterdir() if run_dir.is_dir()]
    logging.info(f"Found {len(catgt_run_paths)} CatGT run dirs: {catgt_run_paths}")
    if not catgt_run_paths:
        raise ValueError("Found no CatGT run dirs to process.")

    for catgt_run_path in catgt_run_paths:
        logging.info(f"Processing CatGT run dir: {catgt_run_path}")

        meta_path = find_one(meta_pattern, parent=catgt_run_path)
        logging.info(f"Parsing metadata from .meta file: {meta_path}")
        metadata = parse_meta(meta_path)
        logging.info(f"Parsed metadata: {metadata}")

        data_run_path = Path(data_path, catgt_run_path.name)
        logging.info(f"Looking for binary data in run dir: {data_run_path}")
        bin_path = find_one(bin_pattern, parent=data_run_path)
        channel_count = metadata[channel_count_meta_name]
        logging.info(f"Loading voltages from .bin file: {bin_path}")
        logging.info(f"Taking channel index {channel_index} from channel count {channel_count}.")
        raw_max = metadata[raw_max_meta_name]
        voltage_max = metadata[voltage_max_meta_name]
        voltage_min = metadata[voltage_min_meta_name]
        channel_voltages = load_voltages(
            bin_path,
            channel_count,
            channel_index,
            raw_max,
            voltage_max,
            voltage_min,
        )
        sample_count = len(channel_voltages)
        logging.info(f"Loaded {sample_count} volatages from {channel_voltages.min()} to {channel_voltages.max()}.")

        sample_rate = metadata[sample_rate_meta_name]
        raw_times = np.arange(sample_count) / sample_rate
        logging.info(f"Using raw sample times at {sample_rate}Hz ranging from {raw_times.min()} to {raw_times.max()}")

        from_sync_path = find_one(from_sync_pattern, parent=catgt_run_path)
        logging.info(f"Loading FROM sync events: {from_sync_path}")
        from_sync = np.loadtxt(from_sync_path)
        logging.info(f"Loaded {len(from_sync)} FROM sync events ranging from {from_sync.min()} to {from_sync.max()}")

        to_sync_path = find_one(to_sync_pattern, parent=catgt_run_path)
        logging.info(f"Loading TO sync events: {to_sync_path}")
        to_sync = np.loadtxt(to_sync_path)
        logging.info(f"Loaded {len(to_sync)} TO sync events ranging from {to_sync.min()} to {to_sync.max()}")

        aligned_times = map_events(
            raw_times,
            from_sync,
            to_sync
        )
        logging.info(f"Aligned {len(aligned_times)} sample times from {aligned_times.min()} to {aligned_times.max()}")

        output_run_path = Path(output_path, catgt_run_path.name)
        output_run_path.mkdir(exist_ok=True, parents=True)
        voltage_out_name = Path(output_run_path, f"{channel_name}_voltage.npy")
        logging.info(f"Writing {channel_name} channel voltages to: {voltage_out_name}")
        np.save(voltage_out_name, channel_voltages)

        times_out_name = Path(output_run_path, f"{channel_name}_times.txt")
        logging.info(f"Writing {channel_name} channel times to: {times_out_name}")
        np.savetxt(times_out_name, aligned_times, fmt='%.6f', delimiter='\t')


def main(argv: Optional[Sequence[str]] = None) -> int:
    set_up_logging()

    parser = ArgumentParser(description="Align continuous signal sample times to a destination clock.")
    parser.add_argument(
        "data_root",
        type=str,
        help="directory with raw data from one or more runs"
    )
    parser.add_argument(
        "catgt_root",
        type=str,
        help="directory with CatGT outputs from one or more runs"
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="directory to write adjusted signals from each run."
    )
    parser.add_argument(
        "--bin-pattern",
        type=str,
        help='Glob pattern for finding a raw binary file with a channel to convert and align, within each run subdir of DATA_ROOT. (default: %(default)s)',
        default="*.bin"
    )
    parser.add_argument(
        "--meta-pattern",
        type=str,
        help="Glob pattern for finding a SpikeGlx .meta file that describes the binary file from BIN_PATTERN, within each run subdir of CATGT_ROOT. (default: %(default)s)",
        default="*.meta"
    )
    parser.add_argument(
        "--from-sync-pattern",
        type=str,
        help="Glob pattern for finding a text file of FROM stream sync pulse times, within each run subdir of CATGT_ROOT. (default: %(default)s)",
        default="*xd_13_6_500.txt"
    )
    parser.add_argument(
        "--to-sync-pattern",
        type=str,
        help="Glob pattern for finding a text file of TO stream sync pulse times, within each run subdir of CATGT_ROOT. (default: %(default)s)",
        default="*/*imec0.ap.*.txt"
    )
    parser.add_argument(
        "--channel-count-meta-name",
        type=str,
        help="Name of the channel count property within each meta file from META_PATTERN. (default: %(default)s)",
        default="nSavedChans"
    )
    parser.add_argument(
        "--voltage-max-meta-name",
        type=str,
        help="Name of the max voltage property within each meta file from META_PATTERN. (default: %(default)s)",
        default="obAiRangeMax"
    )
    parser.add_argument(
        "--voltage-min-meta-name",
        type=str,
        help="Name of the min voltage property within each meta file from META_PATTERN. (default: %(default)s)",
        default="obAiRangeMin"
    )
    parser.add_argument(
        "--raw-max-meta-name",
        type=str,
        help="Name of the max raw value property within each meta file from META_PATTERN. (default: %(default)s)",
        default="obMaxInt"
    )
    parser.add_argument(
        "--sample-rate-meta-name",
        type=str,
        help="Name of the sample rate property within each meta file from META_PATTERN. (default: %(default)s)",
        default="obSampRate"
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        help="Index of which channel to read from each raw binary file from BIN_PATTERN. (default: %(default)s)",
        default=7
    )
    parser.add_argument(
        "--channel-name",
        type=str,
        help="Name to use for output data files with voltage and aligned sample times, for each signal from CHANNEL_INDEX. (default: %(default)s)",
        default="treadmill"
    )

    cli_args = parser.parse_args(argv)

    data_path = Path(cli_args.data_root)
    catgt_path = Path(cli_args.catgt_root)
    output_path = Path(cli_args.output_root)
    try:
        align_signal(
            data_path,
            catgt_path,
            output_path,
            cli_args.meta_pattern,
            cli_args.bin_pattern,
            cli_args.from_sync_pattern,
            cli_args.to_sync_pattern,
            cli_args.channel_count_meta_name,
            cli_args.voltage_max_meta_name,
            cli_args.voltage_min_meta_name,
            cli_args.raw_max_meta_name,
            cli_args.sample_rate_meta_name,
            cli_args.channel_index,
            cli_args.channel_name,
        )
    except:
        logging.error("Error aligning continuous signal times.", exc_info=True)
        return -1

    return 0


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)
