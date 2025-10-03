# Kafka Pilot Developer Guide

This guide explains how to run the Kafka pilot locally and in CI.

Prerequisites
- Docker and Docker Compose
- Python 3.11
- Recommended: a virtualenv

Client runtime notes
- This project uses `kafka-python` (Apache-2.0) as the Kafka client in the pilot. It does not use third-party vendor client libraries. The code includes compatibility detection for the common registry wire format (magic byte + schema id) so we can interoperate with systems that use that format, but we do not depend on vendor-provided code or libraries.

Local steps
1. Install Python dependencies
   ```bash
   python -m pip install -r kafka/requirements.txt
   ```

2. Start Kafka dev stack
   ```bash
   docker-compose -f kafka/docker/docker-compose.kafka.yml up -d --build
   ```

3. Bootstrap topics and schemas
   ```bash
   python kafka/scripts/bootstrap_pilot.py --bootstrap localhost:9092 --registry http://localhost:8081
   ```

4. Run unit tests
   ```bash
   pytest -q kafka/tests
   ```

5. Run integration test locally
   ```bash
   RUN_PILOT_INTEGRATION=1 KAFKA_BOOTSTRAP_SERVERS=localhost:9092 SCHEMA_REGISTRY_URL=http://localhost:8081 pytest -q kafka/tests/test_integration_pilot.py
   ```

6. Fast in-process integration test (no Docker required)

   For rapid feedback during development you can run a fast in-process integration test
   that uses the in-memory MpcAdapter (no Kafka/Schema Registry required):

   ```bash
   pytest -q kafka/tests/test_inprocess_integration.py
   ```

   This test exercises the Scout -> Crawler -> Memory flow entirely in-process and is
   intended for developer iteration.

7. Compatibility enforcement in bootstrap

   The bootstrap script (`kafka/scripts/bootstrap_pilot.py`) will register schemas
   and set subject compatibility to BACKWARD for the pilot subjects by default.

Notes on Schema Registry choice

- This project uses Apicurio Registry by default in the dev docker-compose stack. Apicurio is Apache-licensed and provides a native Registry API (v2) and a ccompat compatibility layer at `/apis/ccompat/v6`.
- The bootstrap script will register schemas against the registry endpoint and set per-subject compatibility to BACKWARD by default. If you run another registry implementation the bootstrap attempts known registry endpoints (Apicurio and Karapace-compatible root API) for compatibility.

Useful endpoints:
- Apicurio ccompat API: http://localhost:8081/apis/ccompat/v6
  - Register: POST /apis/ccompat/v6/subjects/{subject}/versions
  - Get by id: GET /apis/ccompat/v6/ids/{id}
  - Set compatibility: PUT /apis/ccompat/v6/config/{subject}
- Apicurio native Registry API: http://localhost:8081/apis/registry/v2
  - Register: POST /apis/registry/v2/subjects/{subject}/versions
  - Get by id: GET /apis/registry/v2/ids/{id}
- Karapace / generic registry-compatible root API:
  - Register: POST /subjects/{subject}/versions
  - Get by id: GET /schemas/ids/{id}
  - Set compatibility: PUT /config/{subject}

Notes
- The bootstrap script registers the latest schema file found for a topic as the subject `<topic>-value` in Schema Registry.
- CI runs a full boot-up, registers schemas and runs unit + the integration test.
- The CI is split into two jobs:
   - `unit-tests`: runs fast unit and in-process integration tests without starting Docker.
   - `integration-tests`: boots the Kafka dev stack, registers schemas, and runs the full integration test.
- The CI uses pip caching to speed up installs. The kafka-python client path does not require librdkafka.
