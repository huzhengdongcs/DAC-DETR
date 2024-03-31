#!/usr/bin/env bash

set -x

EXP_DIR=output/r50_deformplus_cdn_ice/12eps/results/
PY_ARGS=${@:1}

python3 -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --dim_feedforward 2048 \
    --num_queries_one2one 900 \
    --epochs 12 \
    --lr_drop 11 \
    --dropout 0.0 \
    --mixed_selection \
    --look_forward_twice \
    ${PY_ARGS}
