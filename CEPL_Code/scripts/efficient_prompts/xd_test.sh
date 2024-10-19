#!/bin/bash

# custom config
TRAINER=EFF_Prompts
OUTPUT="path to outputs"
DATA="path to DATA"


DATASET=$1
EPOCH=$2
SEED=$3


NCTX=32  # number of context tokens
CFG=vit_b16_cepl
SHOTS=16
PREC="fp32"


if [[ "$DATASET" == "imagenet" || "$DATASET" == "sun397" ]]; then 

    DIR=${OUTPUT}/evaluation/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/EFF_LR/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${OUTPUT}/CEPL/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED} \
        --load-epoch ${EPOCH} \
        --eval-only \
        TRAINER.EFF_PROMPTS.N_CTX ${NCTX} \
        TRAINER.EFF_PROMPTS.PREC ${PREC}
    fi

else

    DIR=${OUTPUT}/evaluation/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/EFF_Prompts/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${OUTPUT}/CEPL/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED} \
        --load-epoch ${EPOCH} \
        --eval-only \
        TRAINER.EFF_PROMPTS.N_CTX ${NCTX} \
        TRAINER.EFF_PROMPTS.PREC ${PREC}
    fi

fi