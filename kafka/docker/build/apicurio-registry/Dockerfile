FROM eclipse-temurin:17-jre-jammy
ARG APICURIO_VERSION=2.1.0.Final
ARG APICURIO_TAR=apicurio-registry-${APICURIO_VERSION}-deployment.zip

WORKDIR /opt/apicurio
ADD ../../sources/apicurio/${APICURIO_TAR} ./
RUN apt-get update && apt-get install -y unzip && unzip ${APICURIO_TAR} && rm ${APICURIO_TAR}

ENV APICURIO_HOME=/opt/apicurio
EXPOSE 8081
CMD ["/opt/apicurio/bin/run.sh"]
