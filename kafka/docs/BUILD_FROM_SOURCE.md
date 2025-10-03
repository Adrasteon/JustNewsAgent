BUILD FROM SOURCE - Reproducible Image Builds

Overview
This guide explains how to build the runtime images used by the kafka
scaffold from upstream source tarballs or source repositories so the
project does not rely upon opaque vendor images.

Requirements
- podman (preferred) or Docker
- yq (for YAML parsing in scripts)
- sha256sum

Steps
1. Review and update checksums in `kafka/docker/sources/sources.yaml`.
2. Run `./kafka/scripts/build_images.sh` to fetch sources and build images.
3. Use `podman run` or the systemd templates in `kafka/docs/PODMAN.md` to
   run containers as services.

Notes
- If you prefer a hermetic build, consider using Nix or Guix for exact
  dependency control.
- For SeaweedFS we build from the Git repo; for binary distributions we
  store tarballs under `kafka/docker/sources/` with pinned checksums.

Object store
- SeaweedFS is the default and recommended S3-like object store for the
  kafka scaffold. The build-from-source flow builds SeaweedFS from the
  pinned Git tag defined in `kafka/docker/sources/sources.yaml`.
  Avoid using heavier or non-drop-in fallbacks unless operationally
  necessary; discuss first before adding alternatives.

Populate checksums and validation
- Use `kafka/scripts/fetch_and_update_shas.sh` to download upstream tarballs and compute SHA256 checksums.
- Example: `./kafka/scripts/fetch_and_update_shas.sh --update` will update `kafka/docker/sources/sources.yaml` in-place with computed checksums.
- A CI workflow `validate-sources.yml` runs on PRs touching `kafka/docker/sources/` and validates that checksums are present and correct. This prevents accidental inclusion of unverified sources.
