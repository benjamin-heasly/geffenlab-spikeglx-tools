# geffenlab-spikeglx-tools
Docker image build with CatGT, TPrime, and CWaves

Here's how I'm testing this locally, for now.

```
cd geffenlab-spikeglx-tools
./build

docker run -ti --rm geffenlab/geffenlab-spikeglx-tools:local conda_run python /opt/code/catgt.py --help
docker run -ti --rm geffenlab/geffenlab-spikeglx-tools:local conda_run python /opt/code/tprime.py --help
```
