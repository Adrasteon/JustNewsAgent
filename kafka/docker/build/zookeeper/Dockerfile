FROM openjdk:11-jre-slim
ARG ZK_VERSION=3.8.1
ARG ZK_TGZ=zookeeper-${ZK_VERSION}.tar.gz

WORKDIR /opt/zookeeper
ADD ../../sources/zookeeper/${ZK_TGZ} ./
RUN tar -xzf ${ZK_TGZ} --strip-components=1 && rm ${ZK_TGZ}

ENV ZK_HOME=/opt/zookeeper
ENV PATH="$ZK_HOME/bin:$PATH"

EXPOSE 2181

CMD ["/opt/zookeeper/bin/zkServer.sh", "start-foreground"]
