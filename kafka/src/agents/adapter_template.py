"""Adapter template for porting JustNews agents to Kafka producers/consumers.

This module defines a minimal TransportAdapter interface and a lightweight
factory that returns an adapter based on the TRANSPORT environment
variable. Adapters are intentionally small and only wire event production
and consumption; business logic remains in the agent's core code which
can be ported incrementally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import os
import logging

from .schema_registry import SchemaRegistryClient

logger = logging.getLogger(__name__)


@dataclass
class EventEnvelope:
    event_id: str
    event_type: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any]


class TransportAdapter:
    """Abstract adapter interface used by agents.

    Implementations should be side-effect free when possible and return
    explicit result objects to simplify testing.
    """

    def produce(self, topic: str, envelope: EventEnvelope) -> None:
        raise NotImplementedError()

    def consume(self, topic: str, handler) -> None:
        """Register a handler callable(topic, envelope) for incoming events.

        Implementations may run the consumer loop in the foreground or in
        a separate thread/process depending on runtime configuration.
        """
        raise NotImplementedError()


class MpcAdapter(TransportAdapter):
    """Legacy MCP adapter placeholder.

    In the scaffold we provide a minimal shim that records events to a
    local in-memory queue for testing.
    """

    def __init__(self):
        self._queue = []
        self._handlers = {}

    def produce(self, topic: str, envelope: EventEnvelope) -> None:
        # If a handler is registered for this topic, deliver immediately.
        handler = self._handlers.get(topic)
        if handler:
            try:
                handler(topic, envelope)
                return
            except Exception:
                # If handler fails, fall back to queueing the message
                logger.exception("In-memory handler failed for topic %s", topic)
        # Otherwise append to queue for later consumers
        self._queue.append((topic, envelope))

    def consume(self, topic: str, handler) -> None:
        # Register handler and replay any queued messages for this topic
        self._handlers[topic] = handler
        for t, env in list(self._queue):
            if t == topic:
                try:
                    handler(topic, env)
                except Exception:
                    logger.exception("Handler raised while replaying queued message for topic %s", topic)


class KafkaAdapter(TransportAdapter):
    """Lightweight Kafka adapter that uses kafka-python and fastavro when available.

    Behavior:
    - If `kafka-python` is installed, create a Producer and Consumer and provide
      basic produce() and consume() implementations.
    - If not installed, fall back to the existing in-memory recording behavior.
    """

    def __init__(self, bootstrap_servers: Optional[str] = None, schema_registry: Optional[str] = None):
        self.bootstrap_servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
        self.schema_registry = schema_registry or os.getenv("SCHEMA_REGISTRY_URL", os.getenv("KAFKA_SCHEMA_REGISTRY", "http://localhost:8081"))
        self._handlers = {}
        self._consumer_thread = None
        self._running = False

        # Try to import kafka-python and fastavro — if missing, log and fallback to in-memory behavior
        try:
            from kafka import KafkaProducer, KafkaConsumer
            import fastavro
            self._ProducerImpl = KafkaProducer
            self._ConsumerImpl = KafkaConsumer
            self._fastavro = fastavro
            self._use_real = True
        except Exception:
            logger.warning("kafka-python or fastavro not available; KafkaAdapter will use in-memory fallback")
            self._ProducerImpl = None
            self._ConsumerImpl = None
            self._fastavro = None
            self._use_real = False
            self._queue = []

        if self._use_real:
            # Create producer with kafka-python
            self._producer = self._ProducerImpl(bootstrap_servers=self.bootstrap_servers)
            # Consumer configuration template for kafka-python
            self._consumer_conf = {
                "bootstrap_servers": self.bootstrap_servers,
                "group_id": os.getenv("KAFKA_CONSUMER_GROUP", "justnews-pilot-group"),
                "auto_offset_reset": "earliest",
            }

    def _topic_base_name(self, topic: str) -> str:
        """Return the base schema name for a topic (e.g., scout.article.created -> article.created)."""
        parts = topic.split('.')
        if len(parts) >= 3:
            return '.'.join(parts[1:3]) if parts[0] == 'scout' else '.'.join(parts[-2:])
        return topic

    def _list_schema_candidates(self, base: str):
        schemas_dir = os.path.join(os.path.dirname(__file__), '..', 'config', 'schemas')
        schemas_dir = os.path.abspath(schemas_dir)
        try:
            files = os.listdir(schemas_dir)
        except Exception:
            return []
        candidates = [f for f in files if f.startswith(base) and (f.endswith('.avsc') or f.endswith('.json'))]
        candidates.sort()
        return [os.path.join(schemas_dir, c) for c in candidates]

    def _select_latest_schema(self, base: str):
        candidates = self._list_schema_candidates(base)
        return candidates[-1] if candidates else None

    def _serialize_avro(self, schema_path: str, payload: dict, schema_id: Optional[int] = None) -> bytes:
        import io, json as _json
        with open(schema_path, 'r', encoding='utf-8') as fh:
            avro_schema = _json.load(fh)
        bio = io.BytesIO()
        self._fastavro.schemaless_writer(bio, avro_schema, payload)
        payload_bytes = bio.getvalue()
        if schema_id is not None:
            # Registry wire format (magic byte + 4-byte schema id)
            import struct
            return b"\x00" + struct.pack('>I', int(schema_id)) + payload_bytes
        return payload_bytes

    def _deserialize_avro(self, topic: str, val: bytes):
        import io, json as _json, struct
        schema_path = self._select_latest_schema(self._topic_base_name(topic))
        with open(schema_path, 'r', encoding='utf-8') as fh:
            avro_schema = _json.load(fh)
        # Detect registry wire format (magic byte + 4-byte schema id)
        if val and len(val) > 5 and val[0] == 0x00:
            schema_id = struct.unpack('>I', val[1:5])[0]
            try:
                sr = SchemaRegistryClient(self.schema_registry)
                schema_text = sr.get_schema_by_id(schema_id)
                if schema_text:
                    avro_schema = _json.loads(schema_text)
                    return self._fastavro.schemaless_reader(io.BytesIO(val[5:]), avro_schema)
            except Exception:
                # Fall back to local schema
                return self._fastavro.schemaless_reader(io.BytesIO(val[5:]), avro_schema)
        # Not registry wire format
        return self._fastavro.schemaless_reader(io.BytesIO(val), avro_schema)

    def _handle_message(self, topic: str, msg_value: bytes, msg_key: Optional[bytes]):
        # Attempt to decode with local schema if available
        payload = None
        schema_path = self._select_latest_schema(self._topic_base_name(topic))
        if schema_path and schema_path.endswith('.avsc') and self._fastavro is not None:
            try:
                payload = self._deserialize_avro(topic, msg_value)
            except Exception:
                payload = msg_value
        else:
            try:
                import json as _json
                payload = _json.loads(msg_value.decode('utf-8'))
            except Exception:
                payload = msg_value

        handler = self._handlers.get(topic)
        if handler:
            env = EventEnvelope(event_id=msg_key.decode('utf-8') if msg_key else '', event_type=topic, payload=payload, metadata={})
            try:
                handler(topic, env)
            except Exception:
                logger.exception('Handler raised an exception for topic %s', topic)

    def produce(self, topic: str, envelope: EventEnvelope) -> None:
        if not self._use_real:
            # In-memory fallback: if a handler is registered for this topic,
            # deliver immediately (same behavior as MpcAdapter). Otherwise
            # append to the queue for later replay.
            handler = self._handlers.get(topic)
            if handler:
                try:
                    handler(topic, envelope)
                    return
                except Exception:
                    logger.exception("In-memory handler failed for topic %s", topic)
            self._queue.append((topic, envelope))
            return

        # Serialize envelope.payload using Avro/JSON based on available schema
        schema_path = self._topic_to_schema_path(topic)
        value_bytes = None
        # headers placeholder kept for future use if needed
        headers = None
        if schema_path and schema_path.endswith('.avsc') and self._fastavro is not None:
            # If schema registry is reachable, register and use registry wire format (magic byte + schema id)
            try:
                sr = SchemaRegistryClient(self.schema_registry)
                with open(schema_path, 'r', encoding='utf-8') as fh:
                    schema_str = fh.read()
                subject = f"{topic}-value"
                schema_id = sr.register_schema(subject, schema_str)
            except Exception:
                schema_id = None

            try:
                payload_bytes = self._serialize_avro(schema_path, envelope.payload, schema_id)
            except Exception as e:
                logger.exception("Failed to Avro-serialize payload for topic %s: %s", topic, e)
                raise
            value_bytes = payload_bytes
        else:
            # Fallback to JSON bytes
            import json as _json
            value_bytes = _json.dumps(envelope.payload).encode('utf-8')

        # Produce to Kafka using kafka-python
        # kafka-python expects bytes for key/value
        self._producer.send(topic, value=value_bytes, key=envelope.event_id.encode('utf-8'))
        self._producer.flush()

    def consume(self, topic: str, handler) -> None:
        # Register handler; start the consumer loop if using real Kafka
        self._handlers[topic] = handler
        if not self._use_real:
            # Replay any in-memory messages matching this topic
            for t, env in list(self._queue):
                if t == topic:
                    handler(t, env)
            return

        # Start background consumer loop when the first handler is registered
        if self._consumer_thread is None:
            from threading import Thread

            def _loop(adapter: 'KafkaAdapter'):
                # kafka-python consumer takes topic list and config
                topics = list(adapter._handlers.keys())
                consumer = adapter._ConsumerImpl(topics, **adapter._consumer_conf)
                adapter._running = True
                try:
                    while adapter._running:
                        msg_pack = consumer.poll(timeout_ms=1000)
                        if not msg_pack:
                            continue
                        for _tp, messages in msg_pack.items():
                            for m in messages:
                                t = m.topic
                                val = m.value
                                key = m.key
                                adapter._handle_message(t, val, key)
                finally:
                    try:
                        consumer.close()
                    except Exception:
                        pass

            self._consumer_thread = Thread(target=_loop, args=(self,), daemon=True)
            self._consumer_thread.start()


def get_adapter() -> TransportAdapter:
    transport = os.getenv("TRANSPORT", "mpc")
    if transport == "kafka":
        return KafkaAdapter()
    return MpcAdapter()


def get_object_store_endpoint() -> str:
    """Return the object store endpoint used by agents.

    Priority:
    1. Environment variable OBJECT_STORE_ENDPOINT
    2. Default: http://localhost:8333 (SeaweedFS default in the scaffold)
    """
    return os.getenv("OBJECT_STORE_ENDPOINT", "http://localhost:8333")


def get_object_store_client() -> object:
    """Return an object-store client instance based on OBJECT_STORE_TYPE.

    Supported values:
    - 'seaweed' (default): use SeaweedFSClient placeholder (local file writes in scaffold)
    - 'ipfs': use a local IPFS daemon via IpfsClient

    The factory prefers SeaweedFS by default to avoid external dependencies
    while providing an IPFS option for decentralized snapshotting where
    the operational team runs an IPFS node.
    """
    store_type = os.getenv("OBJECT_STORE_TYPE", "seaweed").lower()
    if store_type == "ipfs":
        try:
            from .ipfs_client import IpfsClient

            return IpfsClient()
        except Exception:
            # If IPFS isn't available, fall back to SeaweedFS placeholder
            logger.warning("IPFS client unavailable; falling back to SeaweedFSClient")
            return SeaweedFSClient()
    # Prefer SeaweedFS HTTP client when present
    try:
        from .seaweedfs_client import SeaweedFSHttpClient

        return SeaweedFSHttpClient()
    except Exception:
        logger.debug("SeaweedFSHttpClient not available; using placeholder SeaweedFSClient")
        return SeaweedFSClient()


class SeaweedFSClient:
    """Lightweight SeaweedFS client placeholder used by agents in the
    kafka scaffold. This implementation is intentionally simple and
    dependency-free: it simulates uploads and returns a deterministic
    object key. Replace with a real HTTP S3-compatible client for
    production usage.
    """

    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint or get_object_store_endpoint()

    def upload_bytes(self, data: bytes, key_hint: str = "obj") -> str:
        """Upload bytes to SeaweedFS and return an object key.

        This placeholder writes the data to a local temporary file inside
        the scaffold under kafka/tmp/ for development testing and returns
        a pseudo key. In production, replace this method with an S3/RGW
        compatible HTTP PUT or use an SDK.
        """
        tmpdir = os.path.join(os.path.dirname(__file__), "..", "tmp")
        try:
            os.makedirs(tmpdir, exist_ok=True)
        except Exception:
            pass
        # Create deterministic filename based on hash
        import hashlib

        digest = hashlib.sha256(data).hexdigest()[:16]
        filename = f"{key_hint}-{digest}.bin"
        path = os.path.join(tmpdir, filename)
        with open(path, "wb") as f:
            f.write(data)
        # Return a pseudo object key that other scaffold components can use
        return f"seaweed://{filename}"
