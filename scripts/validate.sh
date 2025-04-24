set -x

CONFIG=$1
CKPT=$2
EXPID=${3:-"test_unet"} # optional



python ./scripts/validate.py \
    --cfg ${CONFIG} \
    --valid-batch 1 \
    --checkpoint ${CKPT} \
    --exp-id ${EXPID}
