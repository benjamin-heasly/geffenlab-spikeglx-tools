# geffenlab-spikeglx-tools

Bundle SpikeGLX tools CatGT, TPrime, and CWaves with convenient Python wrappers.


# Building Docker image versions

This repo is configured to automatically build and publish a new Docker image version, each time a [repo tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) is pushed to GitHub.

## Published versions

The published images are located in the GitHub Container Registry as [geffenlab-spikeglx-tools](https://github.com/benjamin-heasly/geffenlab-spikeglx-tools/pkgs/container/geffenlab-spikeglx-tools).  You can find the latest published version at this page.

You can access published images using their full names.  For version `v0.0.0` the full name would be `ghcr.io/benjamin-heasly/geffenlab-spikeglx-tools:v0.0.0`.  You can use this name in [Nexflow pipeline configuration](https://github.com/benjamin-heasly/geffenlab-ephys-pipeline/blob/master/pipeline/main.nf#L3) and with Docker commands like:

```
docker pull ghcr.io/benjamin-heasly/geffenlab-spikeglx-tools:v0.0.0
```

## Releasing new versions

Here's a workflow for building and realeasing a new Docker image version.

First, make changes to the code in this repo, and `push` the changes to GitHub.

```
# Edit code
git commit -a -m "Now with lasers!"
git push
```

Next, create a new repository [tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging), which marks the most recent commit as important, giving it a unique name and description.

```
# Review existing tags and choose the next version number to use.
git pull --tags
git tag -l

# Create the tag for the next version
git tag -a v0.0.5 -m "Now with lasers!"
git push --tags
```

GitHub should automatically kick off a build and publish workflow for the new tag.
You can follow the workflow progress at the repo's [Actions](https://github.com/benjamin-heasly/geffenlab-spikeglx-tools/actions) page.

You can see the workflow code in [build-tag.yml](./.github/workflows/build-tag.yml).
