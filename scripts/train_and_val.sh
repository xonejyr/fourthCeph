set -x  # 启用命令回显，执行时打印每条命令及其参数（调试用）

CONFIG=$1   # necessary
EXPID=$2    # optional

# train with metrics on val dataset
./scripts/train_1_1.sh ${CONFIG} ${EXPID}

# validate with metrics on test dataset
CONFIG_NAME="${CONFIG##*/}" 
CONFIG_PREFIX="${CONFIG_NAME%%.*}"
 ./scripts/validate.sh ${CONFIG} ./exp/${EXPID}-${CONFIG_PREFIX}/best.pth ${EXPID}

# get the basic data
./scripts/utils/run_analysis.sh ${CONFIG} ${EXPID}