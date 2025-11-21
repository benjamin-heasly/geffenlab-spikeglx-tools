import sys
from os import environ
from argparse import ArgumentParser
from typing import Optional, Sequence, Any
import logging
from pathlib import Path
from shutil import copy2, copytree
import subprocess

import numpy as np

from files import find_one

tprime_version = environ.get("TPRIME_VERSION", "unknown/local")


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
    logging.info(f"TPrime version: {tprime_version}")


def load_phy_params(
    params_path: Path
) -> dict[str, Any]:
    """Phy params.py is a Python script with parameter assignments.  Evaluate it to get a dictionary of parameters."""
    logging.info(f"Reading Phy params.py: {params_path}")
    exec(params_path.read_text())
    return locals()


def phy_spike_times_to_seconds(
    params_py: Path,
    spike_times_samples: Path,
    spike_times_seconds: Path
) -> tuple[float, Path]:
    """Convert Phy's spike_times.npy from sample numbers to seconds, using sample_rate from Phy's params.py."""
    # Load params.py so we can get the spikes sample_rate.
    phy_params = load_phy_params(params_py)
    sample_rate = phy_params["sample_rate"]
    logging.info(f"Found Phy sample_rate: {sample_rate}")

    # Convert spike times to seconds so that TPrime can adjust them.
    logging.info(f"Converting spike times from samples to seconds: {spike_times_samples} -> {spike_times_seconds}")
    spike_times = np.load(spike_times_samples)
    spike_times_in_seconds = spike_times / sample_rate
    np.save(spike_times_seconds, spike_times_in_seconds)

    return sample_rate, spike_times_seconds


def phy_spike_times_to_samples(
    params_py: Path,
    spike_times_seconds: Path,
    spike_times_samples: Path,
):
    """Convert adjusted spike times from seconds back to samples, using sample_rate from Phy's params.py."""
    # Load params.py so we can get the spikes sample_rate.
    phy_params = load_phy_params(params_py)
    sample_rate = phy_params["sample_rate"]
    logging.info(f"Found Phy sample_rate: {sample_rate}")

    # Convert spike times in seconds back to samples, so that eg. Phy can work with them.
    spike_times_in_seconds = np.load(spike_times_seconds)
    spike_times_in_samples = np.round(spike_times_in_seconds * sample_rate)

    # Wrtie a new eg spike_times.npy with spike times in seconds.
    np.save(spike_times_samples, spike_times_in_samples.astype(np.int64))


def run_tprime(
    catgt_root: str,
    phy_root: str,
    output_root: str,
    to_stream: str,
    from_streams: list[tuple[str, str]],
    phy_from_stream: str,
    probe_id: str,
    phy_pattern: str,
    sync_period: float,
    offsets_file: str,
    runit_path: str
) -> int:
    """Use TPrime to find and adjust event and spike times, and write adjusted times to the output root."""

    # Start building up TPrime command args.
    catgt_path = Path(catgt_root).absolute()
    to_stream_path = find_one(to_stream, parent=catgt_path)
    tprime_command = [
        runit_path,
        f"-syncperiod={sync_period}",
        f"-tostream={to_stream_path.as_posix()}",
    ]

    # For multi-part recordings, TPrime can read an "offsets" file produced by CatGT.
    if offsets_file is not None:
        offset_arg = f"-offsets={offsets_file}"
        tprime_command.append(offset_arg)

    output_path = Path(output_root).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    spike_times_seconds_adjusted = None
    if phy_from_stream is not None:
        # Locate Phy inputs and choose an output path with the same structure.
        phy_path = Path(phy_root).absolute()
        params_py = find_one(phy_pattern, probe_id, parent=phy_path)
        params_py_relative = params_py.relative_to(phy_path)
        phy_output_path = Path(output_path, "phy", params_py_relative.parent)
        logging.info(f"Writing converted spike times to {phy_output_path}")
        phy_output_path.mkdir(exist_ok=True, parents=True)

        # Make a copy of the original spike_times.npy for reference.
        spike_times_npy = Path(params_py.parent, "spike_times.npy")
        spike_times_original = Path(phy_output_path, "spike_times_original.npy")
        logging.info(f"Copying {spike_times_npy} to {spike_times_original}.")
        copy2(spike_times_npy, spike_times_original)

        # Convert Phy's spike times, in samples, to seconds for TPrime.
        spike_times_seconds = Path(phy_output_path, "spike_times_sec_original.npy")
        phy_spike_times_to_seconds(params_py, spike_times_original, spike_times_seconds)

        # Which clock / time stream / sync events file do spike times come from?
        phy_from_path = find_one(phy_from_stream, parent=catgt_path)
        phy_index = len(from_streams)
        from_stream_arg = f"-fromstream={phy_index},{phy_from_path.as_posix()}"
        tprime_command.append(from_stream_arg)

        # Write adjusted spike times, in seconds, to the outputs dir.
        spike_times_seconds_adjusted = Path(phy_output_path, "spike_times_sec_adj.npy")
        event_arg = f"-events={phy_index},{spike_times_seconds.as_posix()},{spike_times_seconds_adjusted.as_posix()}"
        tprime_command.append(event_arg)

    for index, (from_glob, events_glob) in enumerate(from_streams):
        # Which clock / time stream / sync events file do these events come from?
        from_path = find_one(from_glob, parent=catgt_path)
        from_stream_arg = f"-fromstream={index},{from_path.as_posix()}"
        tprime_command.append(from_stream_arg)

        # Write adjusted events to the output_root.
        events_paths = catgt_path.glob(events_glob)
        for events_path in events_paths:
            events_adjusted = Path(output_path, events_path.relative_to(catgt_path))
            events_adjusted.parent.mkdir(parents=True, exist_ok=True)
            event_arg = f"-events={index},{events_path.as_posix()},{events_adjusted.as_posix()}"
            tprime_command.append(event_arg)

    logging.info(f"Running TPrime with command {tprime_command}")

    result = subprocess.run(tprime_command, check=False, cwd=output_path)
    tprime_log = Path(output_path, "TPrime.log")
    logging.info(f"Reading from TPrime log '{tprime_log}'")
    with open(tprime_log, 'r') as log:
        for line in log:
            print(line)

    if (result.returncode == 0 and spike_times_seconds_adjusted is not None) and spike_times_seconds_adjusted.exists():
        # Write out the spike times adjusted by TPrime (in seconds) as sample numbers for Phy.
        params_py = find_one(phy_pattern, probe_id, parent=phy_path)
        spike_times_adj_npy = Path(spike_times_seconds_adjusted.parent, "spike_times_adj.npy")
        phy_spike_times_to_samples(params_py, spike_times_seconds_adjusted, spike_times_adj_npy)

        # Make a full copy of the input phy/ dir, then replace its spike_times.npy with the adjusted spike_times_adj.npy.
        logging.info(f"Copying contents of phy/ dir from {params_py.parent} to {spike_times_seconds_adjusted.parent}")
        copytree(params_py.parent, spike_times_seconds_adjusted.parent, dirs_exist_ok=True)

        spike_times_npy = Path(params_py.parent, "spike_times.npy")
        logging.info(f"Replacing original spike times in {spike_times_npy} with adjusted {spike_times_adj_npy}")
        copy2(spike_times_adj_npy, spike_times_npy)

    return result.returncode


def from_with_events(value: str):
    parts = value.split(":")
    return (parts[0], parts[1])


def main(argv: Optional[Sequence[str]] = None) -> int:
    set_up_logging()

    parser = ArgumentParser(description="Use TPrime to align event and spike times to a common clock / time stream.")
    parser.add_argument(
        "catgt_root",
        type=str,
        help="directory to search for input event file, like from CatGT"
    )
    parser.add_argument(
        "phy_root",
        type=str,
        help="directory to search for input spike times files, in Phy format"
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="directory to write adjusted event and spike times files."
    )
    parser.add_argument(
        "--to-stream", "-t",
        type=str,
        help="glob to match one text or .npy file with sync event pulses for the destination clock / time stream",
    )
    parser.add_argument(
        "--sync-period", "-s",
        type=float,
        help="nominal period in seconds of sync events (default 1.0s)",
        default=1.0
    )
    parser.add_argument(
        "--from-streams", "-f",
        type=from_with_events,
        nargs="+",
        help='one or more from:events pairs separated by spaces -- each "from" part is a glob to match one text or .npy file with sync event pulses for a source clock / time stream : each "events" part is a pattern for matching other event files, to be converted from the "from" clock to the TO_STREAM clock.  For example: --from-steams ./nidq/sync.txt:/nidq/**/*.txt ./foo/foo.txt:/foo/*.npy',
        default=[]
    )
    parser.add_argument(
        "--phy-from-stream", "-F",
        type=str,
        help="glob to match one text or .npy file with sync event pulses for the Phy spike times source clock / time stream (default None)",
        default=None
    )
    parser.add_argument(
        "--phy-pattern", "-y",
        type=str,
        help='Glob pattern for finding Phy params.py file(s) within PHY_ROOT. (default: %(default)s)',
        default="**/params.py"
    )
    parser.add_argument(
        "--probe-id", "-p",
        type=str,
        help='Probe-specific filter for selecting one params.py among matches for PHY_PATTERN. (default: %(default)s)',
        default="imec0"
    )
    parser.add_argument(
        "--offsets", "-o",
        type=str,
        help="Optional name of CatGT text file with offsets for multiple concatenated recordings. (default: %(default)s)",
        default=None
    )
    parser.add_argument(
        "--runit-path", "-r",
        type=str,
        help="Path to TPrime 'runit.sh' script. (default: %(default)s)",
        default="/opt/TPrime/TPrime-linux/runit.sh"
    )

    cli_args = parser.parse_args(argv)

    try:
        return run_tprime(
            cli_args.catgt_root,
            cli_args.phy_root,
            cli_args.output_root,
            cli_args.to_stream,
            cli_args.from_streams,
            cli_args.phy_from_stream,
            cli_args.probe_id,
            cli_args.phy_pattern,
            cli_args.sync_period,
            cli_args.offsets,
            cli_args.runit_path
        )
    except:
        logging.error("Error running tprime.", exc_info=True)
        return -1


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)
