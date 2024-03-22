#!/bin/bash
#
# Script to play with docker

usage() {
    echo "Usage $(basename $0) <action>: build-image/run/stop/clean-image"
    exit 1
}

CONTAINER_NAME=xinference
IMAGE_NAME=xinference_dev
IMAGE_TMP_NAME=xinference_tmp
CONTAINER_TMP_NAME=xinference_build
# environment
CODE_PATH=$(pwd)/$(dirname $(dirname $(dirname $0)))/../
echo CODE_PATH=$CODE_PATH
export CODE_PATH
SCRIPT_PATH=$CODE_PATH/xinference/deploy/docker

VLLM_NCCL_LIB=${SCRIPT_PATH}/cu12-libnccl.so.2.18.1
VLLM_NCCL_LIB_MD5SUM=296c4de7fbdb0f7fd8501fb63bd0cb40
VLLM_NCCL_URL=https://github.com/vllm-project/vllm-nccl/releases/download/v0.1.0/cu12-libnccl.so.2.18.1

PIP_INDEX=https://pypi.org/simple
PIP_CONF="$HOME/.pip/pip.conf"
if [ -f "$PIP_CONF" ]; then
    USER_PIP_INDEX=$(grep 'index-url=' $PIP_CONF|cut -d= -f 2)
    if [ -n "$USER_PIP_INDEX" ]; then
        PIP_INDEX=$USER_PIP_INDEX
    fi
fi

run_inference_dev_docker() {
    MEM_TOTAL=$(free -g|grep -i mem | awk '{ print $2}')
    mkdir -p ~/xinference_run
    SHM_PARAM=""
    if [[ $MEM_TOTAL -ge 2 ]]; then
        SHM_PARAM="--shm-size=$(($MEM_TOTAL/2))g"
    fi
    echo PARAMS: $SHM_PARAM $@
    docker run --name $CONTAINER_NAME --network host $SHM_PARAM -v "${CODE_PATH}:/opt/inference" -v "${HOME}/xinference_run:/root/.cache" -v "${HOME}/xinference_run:/root/.xinference" -w /opt/inference/ -d $@ $IMAGE_NAME
}

download_nccl() {
    if ! [[ -s $VLLM_NCCL_LIB ]]; then
        /bin/rm -f $VLLM_NCCL_LIB
        echo "Downloading vllm-nccl from $VLLM_NCCL_URL"
        wget "$VLLM_NCCL_URL" -O $VLLM_NCCL_LIB && return
        echo ERROR: Cannot reach github directly. you have to put the file into $VLLM_NCCL_LIB and re-run this script.
        # Cleanup the incomplete file
        /bin/rm $VLLM_NCCL_LIB
        exit 1
    fi
    if [[ -s $VLLM_NCCL_LIB ]]; then
        echo "Verify $VLLM_NCCL_LIB"
        echo "$VLLM_NCCL_LIB_MD5SUM $VLLM_NCCL_LIB" | md5sum -c || exit 1
    fi
}

[ $# -lt 1 ] && usage

case "$1" in
    build-image)
        if (docker image ls $IMAGE_NAME | grep $IMAGE_NAME>/dev/null); then
            echo $IMAGE_NAME already exists.
            echo $0 clean-image-base if you want to rebuild
        else
            docker rm $CONTAINER_TMP_NAME
            docker rmi $IMAGE_TMP_NAME
            download_nccl
            docker build -t $IMAGE_TMP_NAME --build-arg PIP_INDEX=$PIP_INDEX -f ${SCRIPT_PATH}/Dockerfile.dev . || exit 1
            docker create --name $CONTAINER_TMP_NAME --network host -v "${CODE_PATH}:/opt/inference" -w /opt/inference/ $IMAGE_TMP_NAME || exit 1
            docker start $CONTAINER_TMP_NAME
            # Install only the dep, without the code
            docker exec -it $CONTAINER_TMP_NAME pip install -i $PIP_INDEX -e ".[vllm,transformers,embedding,image]"
            docker stop $CONTAINER_TMP_NAME
            image_commit_id=$(docker commit $CONTAINER_TMP_NAME | cut -d: -f2)
            docker tag $image_commit_id $IMAGE_NAME
            docker rm $CONTAINER_TMP_NAME
            docker rmi $IMAGE_TMP_NAME
        fi
        ;;
    run)
        if (docker ps -a | grep $CONTAINER_NAME>/dev/null); then
            docker start $CONTAINER_NAME
        else
            if (which nvidia-smi >/dev/null); then
                run_inference_dev_docker --gpus all
            else
                run_inference_dev_docker
            fi
        fi
        echo "entering $CONTAINER_NAME docker"
        docker exec -it $CONTAINER_NAME bash
        ;;
    stop)
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
        ;;
    clean-image)
        docker rmi $IMAGE_NAME
        ;;
    *)
        usage
        ;;
esac

# vim: set ts=4 sw=4 expandtab:
