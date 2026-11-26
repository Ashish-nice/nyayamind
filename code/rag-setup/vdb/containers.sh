#!/usr/bin/env bash
# Start/stop the local Milvus (embedded etcd + local storage) and Endee servers.
# Both bind to 127.0.0.1 only and keep their data in named Docker volumes.
#   bash code/rag-setup/vdb/containers.sh start-milvus|stop-milvus|start-endee|stop-endee
set -e
HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
export MSYS_NO_PATHCONV=1
wait_http() { for i in $(seq 1 90); do curl -sf -m 3 "$1" >/dev/null 2>&1 && return 0; sleep 4; done; echo "timeout waiting for $1"; return 1; }
case "$1" in
  start-milvus)
    if [ -z "$(docker ps -aq -f name=^nyayamind_milvus$)" ]; then
      docker run -d --name nyayamind_milvus --security-opt seccomp:unconfined --memory 7g \
        -e ETCD_USE_EMBED=true -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
        -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml -e COMMON_STORAGETYPE=local \
        -v nyayamind_milvus:/var/lib/milvus \
        -v "$HERE/docker/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml" \
        -v "$HERE/docker/user.yaml:/milvus/configs/user.yaml" \
        -p 127.0.0.1:19530:19530 -p 127.0.0.1:9091:9091 \
        milvusdb/milvus:v2.5.18 milvus run standalone >/dev/null
    else
      docker start nyayamind_milvus >/dev/null
    fi
    wait_http http://127.0.0.1:9091/healthz && echo "milvus up" ;;
  stop-milvus) docker stop nyayamind_milvus >/dev/null && echo "milvus stopped" ;;
  start-endee)
    if [ -z "$(docker ps -aq -f name=^nyayamind_endee$)" ]; then
      docker run -d --name nyayamind_endee --memory 7g -v nyayamind_endee:/data \
        -p 127.0.0.1:8090:8080 endeeio/endee-server:latest >/dev/null
    else
      docker start nyayamind_endee >/dev/null
    fi
    sleep 5; echo "endee started" ;;
  stop-endee) docker stop nyayamind_endee >/dev/null && echo "endee stopped" ;;
  *) echo "usage: $0 start-milvus|stop-milvus|start-endee|stop-endee"; exit 2 ;;
esac
