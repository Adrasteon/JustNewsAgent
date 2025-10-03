FROM golang:1.20-alpine AS build
ARG SEAWEED_VERSION=2.90
WORKDIR /src
RUN apk add --no-cache git
RUN git clone --depth 1 --branch ${SEAWEED_VERSION} https://github.com/chrislusf/seaweedfs.git /src/seaweedfs
WORKDIR /src/seaweedfs
RUN go build -o weed ./weed

FROM alpine:3.18
COPY --from=build /src/seaweedfs/weed /usr/local/bin/weed
EXPOSE 9333 8333
ENTRYPOINT ["/usr/local/bin/weed"]
