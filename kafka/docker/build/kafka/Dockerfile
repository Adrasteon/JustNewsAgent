# Build Kafka from upstream binary tarball (reproducible build)
# Uses upstream Apache Kafka distribution
FROM openjdk:17-jdk-slim

ARG KAFKA_VERSION=3.5.1
ARG SCALA_VERSION=2.13
ARG KAFKA_TGZ=kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz
ARG KAFKA_URL=https://downloads.apache.org/kafka/${KAFKA_VERSION}/${KAFKA_TGZ}

WORKDIR /opt/kafka

# Download is done at build time by the helper script to preserve reproducibility
# The build context must include the tarball at kafka/docker/sources/kafka/${KAFKA_TGZ}
ADD ../../sources/kafka/${KAFKA_TGZ} ./
RUN tar -xzf ${KAFKA_TGZ} --strip-components=1 && rm ${KAFKA_TGZ}

ENV KAFKA_HOME=/opt/kafka
ENV PATH="$KAFKA_HOME/bin:$PATH"

EXPOSE 9092

COPY server.properties /opt/kafka/config/server.properties

CMD ["/opt/kafka/bin/kafka-server-start.sh", "/opt/kafka/config/server.properties"]
