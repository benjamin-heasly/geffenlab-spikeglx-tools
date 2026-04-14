import sys
from os import environ
from argparse import ArgumentParser
from typing import Optional, Sequence, Any
import logging
from pathlib import Path
from shutil import copy2
import subprocess

import numpy as np

from files import find, find_one


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
    catgt_run_path: Path,
    phy_run_path: Path,
    output_run_path: Path,
    to_sync_path: Path,
    events_sync_path: Path,
    events_paths: list[Path],
    probe_sync_paths: list[Path],
    probe_spikes_paths: list[Path],
    probe_params_paths: list[Path],
    sync_period: float,
    offsets: str,
    runit: str
):
    """Use TPrime to adjust event and spike times, write adjusted events to the output root and adjusted spikes in place."""
    output_run_path.mkdir(parents=True, exist_ok=True)

    # Start building up TPrime command args.
    tprime_command = [
        runit,
        f"-syncperiod={sync_period}",
        f"-tostream={to_sync_path.as_posix()}",
    ]

    # For multi-part recordings, TPrime can read an "offsets" file produced by CatGT.
    if offsets is not None:
        offset_arg = f"-offsets={offsets}"
        tprime_command.append(offset_arg)

    # TPrime keeps track of clocks/streams by index.
    # We'll use one for auxiliary events, plus one per probe.
    stream_index = 0

    # Tell TPrime about auxiliary event sync pulses.
    aux_from_arg = f"-fromstream={stream_index},{events_sync_path.as_posix()}"
    tprime_command.append(aux_from_arg)

    # Tell TPrime which auxiliary events to convert, and where.
    for events_path in events_paths:
        events_relative_path = events_path.relative_to(catgt_run_path)
        events_output_path = Path(output_run_path, events_relative_path)
        events_output_path.parent.mkdir(parents=True, exist_ok=True)
        aux_events_arg = f"-events={stream_index},{events_path.as_posix()},{events_output_path.as_posix()}"
        tprime_command.append(aux_events_arg)

    # Organize Phy files per probe.
    probe_seconds_to_convert = []
    for probe_sync_path, probe_spikes_path, probe_params_path in zip(probe_sync_paths, probe_spikes_paths, probe_params_paths):

        # Declare a new clock/stream for this probe.
        stream_index += 1

        probe_run_path = probe_params_path.parent
        logging.info(f"Aligning spikes for probe: {probe_run_path}")

        probe_run_relative_path = probe_run_path.relative_to(phy_run_path)
        probe_run_output_path = Path(output_run_path, probe_run_relative_path)
        probe_run_output_path.mkdir(exist_ok=True, parents=True)
        logging.info(f"Writing tprime results for probe: {probe_run_output_path}")

        # Make a backup copy of the original spike times, in place.
        # Only do this once, so we don't clobber the backup with an adjusted version of spike_times.npy
        # Going forward we will:
        #   - read original spike times from the backup, spike_times_original.npy
        #   - write adjusted spike times in place of the original, spike_times.npy
        # This approach should make it safe to re-run this pipeline step.
        # Modifying spike_times.npy in place is useful for downstream Phy and Bombcell.
        # We'll write other outputs to a seprate folder, which is generally safer and cleaner.
        probe_spikes_backup_path = probe_spikes_path.with_name("spike_times_original.npy")
        if probe_spikes_backup_path.exists():
            logging.info(f"Found existing spike times backup: {probe_spikes_backup_path}")
        else:
            logging.info(f"Creating spike times backup: {probe_spikes_backup_path}")
            copy2(probe_spikes_path, probe_spikes_backup_path)

        # Phy deals with spike times in sample numbers.  Convert to seconds for TPrime.
        probe_spikes_seconds_path = Path(probe_run_output_path, "spike_times_sec_original.npy")
        phy_spike_times_to_seconds(probe_params_path, probe_spikes_backup_path, probe_spikes_seconds_path)

        # Tell TPrime about sync pulses for this probe.
        probe_from_arg = f"-fromstream={stream_index},{probe_sync_path.as_posix()}"
        tprime_command.append(probe_from_arg)

        # Tell TPrime which spike times to convert, and where.
        probe_spikes_seconds_adjusted_path = Path(probe_run_output_path, "spike_times_sec_adjusted.npy")
        probe_events_arg = f"-events={stream_index},{probe_spikes_seconds_path.as_posix()},{probe_spikes_seconds_adjusted_path.as_posix()}"
        tprime_command.append(probe_events_arg)

        # After TPrime runs, convert the adjusted spike times in seconds back to samples.
        probe_seconds_to_convert.append((probe_spikes_seconds_adjusted_path, probe_spikes_path, probe_params_path))

    logging.info(f"Running TPrime with command {tprime_command}")
    result = subprocess.run(tprime_command, check=False, cwd=output_run_path)
    tprime_log = Path(output_run_path, "TPrime.log")
    logging.info(f"Reading from TPrime log '{tprime_log}'")
    with open(tprime_log, 'r') as log:
        for line in log:
            print(line)

    if result.returncode != 0:
        raise ValueError(f"TPrime exited with nonxero result code {result.returncode}")

    for (probe_spikes_seconds_adjusted_path, probe_spikes_path, probe_params_path) in probe_seconds_to_convert:
        logging.info(f"Updating original spike times in place: {probe_spikes_path}")
        phy_spike_times_to_samples(probe_params_path, probe_spikes_seconds_adjusted_path, probe_spikes_path)


def find_runs_and_align(
    catgt_path: Path,
    phy_path: Path,
    output_path: Path,
    to_sync_pattern: str,
    events_sync_pattern: str,
    events_pattern: str,
    probe_ids: list[str],
    probe_sync_pattern: str,
    probe_spikes_pattern: str,
    probe_params_pattern: str,
    sync_period: float,
    offsets: str,
    runit: str,
) -> int:
    """Locate SpikeGlx/CatGT runs and corresponding sorting results, invoke TPrime for each run in sequence."""

    # Locate SpikeGlx runs as subfolders of the CATGT_ROOT.
    logging.info(f"Processing SpikeGlx/CatGT runs within: {catgt_path}")
    catgt_run_paths = [run_dir for run_dir in catgt_path.iterdir() if run_dir.is_dir()]
    logging.info(f"Found {len(catgt_run_paths)} CatGT run dirs: {catgt_run_paths}")
    if not catgt_run_paths:
        raise ValueError("Found no CatGT run dirs to process.")

    for catgt_run_path in catgt_run_paths:
        logging.info(f"Processing CatGT run dir: {catgt_run_path}")

        to_sync_path = find_one(to_sync_pattern, parent=catgt_run_path)
        logging.info(f"Aligning spikes and event TO desination sync pulses: {to_sync_path}")

        events_sync_path = find_one(events_sync_pattern, parent=catgt_run_path)
        logging.info(f"Aligning auxiliary events FROM sync pulses: {events_sync_path}")

        events_paths = find(events_pattern, parent=catgt_run_path)
        logging.info(f"Aligning auxiliary events FROM {len(events_paths)} original sources: {events_paths}")

        probe_sync_paths = []
        probe_spikes_paths = []
        probe_params_paths = []
        phy_run_path = Path(phy_path, catgt_run_path.name)
        for probe_id in probe_ids:
            logging.info(f"Looking for probe {probe_id} in run dir: {phy_run_path}")

            probe_sync_path = find_one(probe_sync_pattern, filter=probe_id, parent=catgt_run_path, none_ok=True)
            if probe_sync_path is not None:
                logging.info(f"Aligning spikes FROM sync pulses: {probe_sync_path}")
                probe_sync_paths.append(probe_sync_path)

            probe_spikes_path = find_one(probe_spikes_pattern, filter=probe_id, parent=phy_run_path, none_ok=True)
            if probe_spikes_path is not None:
                logging.info(f"Aligning spikes FROM : {probe_spikes_path}")
                probe_spikes_paths.append(probe_spikes_path)

            probe_params_path = find_one(probe_params_pattern, filter=probe_id, parent=phy_run_path, none_ok=True)
            if probe_params_path is not None:
                logging.info(f"Found params.py : {probe_params_path}")
                probe_params_paths.append(probe_params_path)

        output_run_path = Path(output_path, catgt_run_path.name)
        run_tprime(
            catgt_run_path,
            phy_run_path,
            output_run_path,
            to_sync_path,
            events_sync_path,
            events_paths,
            probe_sync_paths,
            probe_spikes_paths,
            probe_params_paths,
            sync_period,
            offsets,
            runit
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    set_up_logging()

    parser = ArgumentParser(description="Use TPrime to align event and spike times to a destination clock.")
    parser.add_argument(
        "catgt_root",
        type=str,
        help="directory with CatGT outputs from one or more runs"
    )
    parser.add_argument(
        "phy_root",
        type=str,
        help="directory with Phy/Kilosort outputs from one or more runs"
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="directory to write adjusted event and spike times from each run."
    )
    parser.add_argument(
        "--to-sync-pattern",
        type=str,
        help="glob to match one text or .npy file of sync pulse times for the destination clock, within CATGT_ROOT, for each run. (default: %(default)s)",
        default="*/*.imec0.ap.*.txt"
    )
    parser.add_argument(
        "--events-sync-pattern",
        type=str,
        help="glob to match one text or .npy file of sync pulse times for auxiliary events (eg nidq or onebox), within CATGT_ROOT, for each run. (default: %(default)s)",
        default="*.nidq.xd_8_4_500.txt"
    )
    parser.add_argument(
        "--events-pattern",
        type=str,
        help="glob to match multiple text or .npy files of auxiliary event times (eg nidq or onebox), within CATGT_ROOT, for each run. (default: %(default)s)",
        default="*.nidq.*.txt"
    )
    parser.add_argument(
        "--probe-ids",
        type=str,
        nargs="+",
        help="One or more probe ids to consider for sorting and for associating CatGT outputs with Kilosort outputs for the same probe. (default: %(default)s)",
        default=["imec0", "imec1"]
    )
    parser.add_argument(
        "--probe-sync-pattern",
        type=str,
        help="glob to match one text or .npy file of sync pulse times per probe, within CATGT_ROOT, for each run. (default: %(default)s)",
        default="*/*.ap.*.txt"
    )
    parser.add_argument(
        "--probe-spikes-pattern",
        type=str,
        help="glob to match one .npy file of spike event times per probe, within PHY_ROOT, for each run. (default: %(default)s)",
        default="*/spike_times.npy"
    )
    parser.add_argument(
        "--probe-params-pattern",
        type=str,
        help="glob to match one params.py file per probe, within PHY_ROOT, for each run. (default: %(default)s)",
        default="*/params.py"
    )
    parser.add_argument(
        "--sync-period",
        type=float,
        help="nominal period in seconds of sync events. (default: %(default)s)",
        default=1.0
    )
    parser.add_argument(
        "--offsets",
        type=str,
        help="Optional name of CatGT text file with offsets for multiple concatenated recordings. (default: %(default)s)",
        default=None
    )
    parser.add_argument(
        "--runit",
        type=str,
        help="Path to TPrime 'runit.sh' script. (default: %(default)s)",
        default="/opt/TPrime/TPrime-linux/runit.sh"
    )

    cli_args = parser.parse_args(argv)

    catgt_path = Path(cli_args.catgt_root)
    phy_path = Path(cli_args.phy_root)
    output_path = Path(cli_args.output_root)
    try:
        find_runs_and_align(
            catgt_path,
            phy_path,
            output_path,
            cli_args.to_sync_pattern,
            cli_args.events_sync_pattern,
            cli_args.events_pattern,
            cli_args.probe_ids,
            cli_args.probe_sync_pattern,
            cli_args.probe_spikes_pattern,
            cli_args.probe_params_pattern,
            cli_args.sync_period,
            cli_args.offsets,
            cli_args.runit,
        )
    except:
        logging.error("Error running tprime.", exc_info=True)
        return -1

    return 0


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)
