import sys
from os import environ
from argparse import ArgumentParser
from typing import Optional, Sequence
import logging
from pathlib import Path
import re
import subprocess
from shutil import copy2


catgt_version = environ.get("CATGT_VERSION", "unknown/local")


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
    logging.info(f"CatGT version: {catgt_version}")


def find_run_and_gate(
    input_path: Path,
    run_gate_delimiter: str
) -> list[tuple[str, str]]:
    """Scan the given input_path as <RUN_NAME>_g<GATE_INDEX>, return the parsed (RUN_NAME, GATE_INDEX)."""
    logging.info(f"Searching input path: {input_path}")
    logging.info(f"Looking for <RUN_NAME>{run_gate_delimiter}<GATE_INDEX>")
    run_name, delimiter, gate_index = input_path.name.partition(run_gate_delimiter)
    if run_name and delimiter and gate_index:
        logging.info(f"Found RUN_NAME {run_name}, GATE_INDEX {gate_index}).")
        return (run_name, gate_index)
    else:
        raise ValueError(f"Could not parse <RUN_NAME>{run_gate_delimiter}<GATE_INDEX> from input path: {input_path.name}")


def find_triggers_and_probes(
    run_path: Path,
    trigger_pattern: str,
    probe_pattern: str,
) -> tuple[set[str], set[str]]:
    """Scan the given run_path for file/dir names like _g0_t<TRIGGER_INDEX> and _imec<PROBE_INDEX>, collect unique TRIGGER_INDEX and PROBE_INDEX."""
    logging.info(f"Searching run path: {run_path}")
    names = [file.name for file in run_path.iterdir()]
    logging.info(f"Found names: {names}")

    logging.info(f"Looking for names with TRIGGER_INDEX like: {trigger_pattern}")
    trigger_matches = [re.search(trigger_pattern, name) for name in names]
    triggers = {match.group(1) for match in trigger_matches if match}
    triggers = list(triggers)
    triggers.sort()
    logging.info(f"Found triggers: {triggers}")

    logging.info(f"Looking for names with PROBE_INDEX like: {probe_pattern}")
    probe_matches = [re.search(probe_pattern, name) for name in names]
    probes = {match.group(1) for match in probe_matches if match}
    probes = list(probes)
    probes.sort()
    logging.info(f"Found probes: {probes}")

    return triggers, probes


def run_catgt(
    input_path: Path,
    output_path: Path,
    run_gate_delimiter: str,
    trigger_pattern: str,
    probe_pattern: str,
    runit_path: str,
    catgt_args: list[str]
):
    logging.info(f"Processing run dir: {input_path}")

    run_name, gate_index = find_run_and_gate(input_path, run_gate_delimiter)
    logging.info(f"Using run name {run_name} and gate index {gate_index}.")

    (triggers, probes) = find_triggers_and_probes(input_path, trigger_pattern, probe_pattern)
    triggers_arg = ",".join(triggers)
    logging.info(f"Using triggers {triggers_arg}")
    probes_arg = ",".join(probes)
    logging.info(f"Using probes {probes_arg}")

    catgt_command = [
        runit_path,
        f"-dest={output_path.absolute()}",
        f"-dir={input_path.parent.absolute()}",
        f"-run={run_name}",
        f"-g={gate_index}",
        f"-t={triggers_arg}",
        f"-prb={probes_arg}"
    ] + catgt_args

    logging.info(f"Running CatGT with command {catgt_command}")

    # CatGT always writes to "CatGT.log", we can't choose the name.
    # Clear any existing log, first.
    catgt_log = Path(output_path, "CatGT.log")
    if catgt_log.exists():
        catgt_log.unlink()

    # Run CatGT and read out the log from this run.
    result = subprocess.run(catgt_command, check=False, cwd=output_path)
    logging.info(f"Reading from CatGT log '{catgt_log}'")
    with open(catgt_log, 'r') as log:
        for line in log:
            print(line)

    # Copy CatGT.log to a run-specific file, for future reference.
    catgt_log_for_subdir = Path(output_path, f"CatGT-{input_path.name}.log")
    copy2(catgt_log, catgt_log_for_subdir)

    logging.info(f"CatGT exited with result code {result.returncode}")
    if result.returncode != 0:
        raise ValueError(f"CatGT exited with nonxero result code {result.returncode}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    set_up_logging()

    parser = ArgumentParser(description="Use CatGT to filter sampled traces and/or extract event times.")
    parser.add_argument(
        "input_root",
        type=str,
        help="Directory containing a SpikeGLX run"
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="Directory to contain CatGT outputs for the run"
    )
    parser.add_argument(
        "--run-gate-delimiter",
        type=str,
        help="Delimiter for partitioning INPUT_ROOT into <RUN_NAME> and <GATE_INDEX>.  (default: %(default)s)",
        default="_g"
    )
    parser.add_argument(
        "--trigger-pattern",
        type=str,
        help="Regular expression for parsing TRIGGER_INDEX out of file names files within INPUT_ROOT.  (default: %(default)s)",
        default="_g\\d+_t(\\d+)"
    )
    parser.add_argument(
        "--probe-pattern",
        type=str,
        help="Regular expression for parsing PROBE_INDEX out of subdir names within INPUT_ROOT.  (default: %(default)s)",
        default="_g\\d+_imec(\\d+)$"
    )
    parser.add_argument(
        "--runit-path",
        type=str,
        help="Path to CatGT 'runit.sh' script.  (default: %(default)s)",
        default="/opt/CatGT/CatGT-linux/runit.sh"
    )

    (cli_args, catgt_args) = parser.parse_known_args(argv)

    input_path = Path(cli_args.input_root)
    output_path = Path(cli_args.output_root)
    output_path.mkdir(exist_ok=True, parents=True)

    try:
        run_catgt(
            input_path,
            output_path,
            cli_args.run_gate_delimiter,
            cli_args.trigger_pattern,
            cli_args.probe_pattern,
            cli_args.runit_path,
            catgt_args
        )
    except:
        logging.error("Error running CatGT.", exc_info=True)
        return -1


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)
