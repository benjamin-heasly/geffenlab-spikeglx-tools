import sys
from os import environ
from argparse import ArgumentParser
import re
from typing import Optional, Sequence
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import subprocess
import json

from files import find_one

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


def run_catgt(
    input_path: Path,
    output_path: Path,
    run_name: str,
    gate: str,
    trigger: str,
    probe_index: str,
    runit_path: str,
    catgt_args: list[str]
):
    catgt_command = [
        runit_path,
        f"-dest={output_path.absolute()}",
        f"-dir={input_path.absolute()}",
        f"-run={run_name}",
        f"-g={gate}",
        f"-t={trigger}",
        f"-prb={probe_index}"
    ] + catgt_args

    logging.info(f"Running CatGT with command {catgt_command}")

    result = subprocess.run(catgt_command, check=False, cwd=output_path)

    catgt_log = Path(output_path, "CatGT.log")
    logging.info(f"Reading from CatGT log '{catgt_log}'")
    with open(catgt_log, 'r') as log:
        for line in log:
            print(line)

    logging.info(f"CatGT exited with result code {result.returncode}")
    if result.returncode != 0:
        raise ValueError(f"CatGT exited with nonxero result code {result.returncode}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    set_up_logging()

    parser = ArgumentParser(description="Use CatGT to filter sampled traces and/or extract event times.")
    parser.add_argument(
        "input_root",
        type=str,
        help="directory containing SpikeGLX run subdirs"
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="directory to contain CatGT outputs for the run"
    )
    parser.add_argument(
        "run",
        type=str,
        help="SpikeGLX run name, for example 'test'"
    )
    parser.add_argument(
        "--gate",
        type=str,
        help="SpikeGLX gate index. (default: %(default)s)",
        default="0"
    )
    parser.add_argument(
        "--trigger",
        type=str,
        help="SpikeGLX trigger index. (default: %(default)s)",
        default="0"
    )
    parser.add_argument(
        "--probe-id",
        type=str,
        help="SpikeGLX probe id, ending in a numeric probe index. (default: %(default)s)",
        default="imec0"
    )
    parser.add_argument(
        "--runit-path",
        type=str,
        help="Path to CatGT 'runit.sh' script, default '/opt/CatGT/CatGT-linux/runit.sh'",
        default="/opt/CatGT/CatGT-linux/runit.sh"
    )

    (cli_args, catgt_args) = parser.parse_known_args(argv)

    input_path = Path(cli_args.input_root)
    output_path = Path(cli_args.output_root)
    output_path.mkdir(exist_ok=True, parents=True)

    probe_index = re.search(r"\d+$", cli_args.probe_id).group()
    try:
        run_catgt(
            input_path,
            output_path,
            cli_args.run,
            cli_args.gate,
            cli_args.trigger,
            probe_index,
            cli_args.runit_path,
            catgt_args
        )
    except:
        logging.error("Error running CatGT.", exc_info=True)
        return -1


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)
