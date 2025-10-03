Running the kafka dev stack with Podman and systemd units

Why Podman
- Podman is fully open-source, rootless-capable and does not require
  Docker Desktop. It can generate systemd unit files for running pods
  as system services.

Quick start
1. Build images: `./kafka/scripts/build_images.sh`
2. Create pod: `podman pod create --name justnews-pod -p 9092:9092 -p 2181:2181 -p 8081:8081 -p 9000:8333 -p 5432:5432`
3. Run containers within the pod (example):
   - `podman run -d --pod justnews-pod --name zk local/justnews-zookeeper:3.8.1`
   - `podman run -d --pod justnews-pod --name kafka local/justnews-kafka:3.5.1`
   - `podman run -d --pod justnews-pod --name object-store local/justnews-seaweedfs:2.90`
4. Generate systemd units (optional):
   - `podman generate systemd --new --files --name justnews-pod`
   - Move generated `.service` files to `/etc/systemd/system/` and `systemctl daemon-reload`

Systemd integration
- Podman will generate unit files that manage containers/pods. Use the
  generated units with `systemctl enable --now container-justnews-pod.service`.

Object store notes
- The scaffold uses SeaweedFS as the canonical object store. If you need
  to use a different object store, configure it manually and update
  `kafka/config/kafka_dev.yaml` accordingly. The project avoids adding
  heavier fallback object store services by default to reduce complexity.

Notes on reproducibility
- Always verify tarball checksums in `kafka/docker/sources/sources.yaml`.
- Rebuild images after any source checksum or Dockerfile changes.
