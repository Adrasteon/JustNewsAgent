"""Bootstrap pilot topics and register schemas with Schema Registry.

This script reads `kafka/config/topics/pilot_topics.yaml` and creates topics and
registers schemas to the configured Schema Registry URL.

Usage:
  python kafka/scripts/bootstrap_pilot.py --bootstrap <kafka-bootstrap> --registry <schema-registry-url>

It requires `kafka-python` and `requests` to be installed in the environment.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Dict

import yaml
from kafka.src.agents.schema_registry import SchemaRegistryClient

logger = logging.getLogger(__name__)


def load_topics(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def create_topics(admin_conf: Dict, topics: Dict):
    from kafka.admin import KafkaAdminClient, NewTopic

    admin = KafkaAdminClient(bootstrap_servers=admin_conf.get('bootstrap.servers'))
    new_topics = []
    for t in topics.get('topics', []):
        nt = NewTopic(name=t['name'], num_partitions=t.get('partitions', 1), replication_factor=t.get('replication', 1))
        new_topics.append(nt)
    try:
        admin.create_topics(new_topics=new_topics, timeout_ms=30000)
        logger.info('Requested creation of topics: %s', [t.name for t in new_topics])
    except Exception as e:
        logger.warning('Topic creation reported an exception (may already exist): %s', e)


def find_schema_for_topic(schemas_dir: str, topic_name: str):
    # Basic mapping: topics like 'scout.article.created' -> subject 'article.created-value'
    parts = topic_name.split('.')
    if len(parts) >= 3:
        base = '.'.join(parts[1:3])
    else:
        base = topic_name
    # Search matching files
    candidates = [f for f in os.listdir(schemas_dir) if f.startswith(base) and (f.endswith('.avsc') or f.endswith('.json'))]
    candidates.sort()
    if not candidates:
        return None
    return os.path.join(schemas_dir, candidates[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', default=os.getenv('KAFKA_BOOTSTRAP', 'localhost:9092'))
    parser.add_argument('--registry', default=os.getenv('SCHEMA_REGISTRY_URL', 'http://localhost:8081'))
    parser.add_argument('--topics-file', default=os.path.join(os.path.dirname(__file__), '..', 'config', 'topics', 'pilot_topics.yaml'))
    args = parser.parse_args()

    topics = load_topics(args.topics_file)
    admin_conf = {'bootstrap.servers': args.bootstrap}
    create_topics(admin_conf, topics)

    schemas_dir = os.path.join(os.path.dirname(__file__), '..', 'config', 'schemas')
    from kafka.src.agents.schema_registry import SchemaRegistryClient
    for t in topics.get('topics', []):
        schema_path = find_schema_for_topic(os.path.abspath(schemas_dir), t['name'])
        if schema_path:
            # Register subject as <topic>-value to follow Schema Registry conventions
            subject = f"{t['name']}-value"
            # Use SchemaRegistryClient which supports Apicurio and Karapace registry endpoints
            sr = SchemaRegistryClient(args.registry)
            with open(schema_path, 'r', encoding='utf-8') as sf:
                schema_text = sf.read()
            schema_id = sr.register_schema(subject, schema_text)
            if schema_id:
                logger.info('Registered subject %s with id %s', subject, schema_id)
                # Set compatibility to BACKWARD by default
                compat_ok = sr.set_subject_compatibility(subject, 'BACKWARD')
                # Report which registry endpoint prefix was used for registration/compatibility
                prefix = sr.get_last_successful_prefix()
                if prefix:
                    logger.info('Schema registry detected endpoint prefix: %s for subject %s', prefix, subject)
                else:
                    logger.info('Schema registry endpoint detection: no known prefix succeeded for subject %s', subject)
                if not compat_ok:
                    logger.warning('Failed to set compatibility for subject %s', subject)
            else:
                logger.warning('Failed to register schema for %s; skipping compatibility set', subject)
        else:
            logger.info('No schema found for topic %s', t['name'])

    logger.info('Bootstrap complete')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
