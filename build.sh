#!/bin/sh

set -e

docker build -f environment/Dockerfile -t ghcr.io/benjamin-heasly/geffenlab-spikeglx-tools:local .
