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

    DIR=${OUTPUT}/CEPL/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED}
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
        DATASET.NUM_SHOTS ${SHOTS}\
        TRAINER.EFF_PROMPTS.N_CTX ${NCTX}\
        TRAINER.EFF_PROMPTS.CL EFF_LR\
        TRAINER.EFF_PROMPTS.PREC ${PREC}\
        OPTIM.MAX_EPOCH ${EPOCH}
    fi

else

    DIR=${OUTPUT}/CEPL/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED}
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
        DATASET.NUM_SHOTS ${SHOTS}\
        TRAINER.EFF_PROMPTS.N_CTX ${NCTX}\
        TRAINER.EFF_PROMPTS.CL EFF_Prompts\
        TRAINER.EFF_PROMPTS.PREC ${PREC}\
        OPTIM.MAX_EPOCH ${EPOCH}
    fi

fi