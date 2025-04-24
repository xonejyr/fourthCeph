set -x  # 启用命令回显，执行时打印每条命令及其参数（调试用）

CONFIG=$1   # necessary
EXPID=${2:-"test_unet"} # optional
MODELDIR=${3:-None}
#GPUID=${3:-0}


python ./scripts/train.py \
    --exp-id ${EXPID} \
    --cfg ${CONFIG} \
    --seed 2333 \
    --model-dir ${MODELDIR}
#    --gpu-id ${GPUID}

# how to use the distributive mode on another machine
# 在另一台机器（IP: 192.168.1.11）执行：
# 
# bash
# Copy
# python train.py \
#     --launcher pytorch \
#     --rank 1 \                    # 从节点排名递增
#     --dist-url tcp://192.168.1.10:12345 \  # 指向主节点IP和端口
#     --exp-id my_exp \
#     --cfg configs/spine.yaml
