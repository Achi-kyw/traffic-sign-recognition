# Arguments: --bgs-path, --templates-path, --out-path, --train_images, --val_images
#!/bin/bash
# This script generates a dataset using the provided background images and templates,
BGS_PATH="$1"
TEMPLATES_PATH="$2"
OUT_PATH="$3"
TRAIN_IMAGES=$4
VAL_IMAGES=$5

python generate_dataset.py \
    --bgs-path ${BGS_PATH} \
    --templates-path  ${TEMPLATES_PATH}\
    --out-path "${OUT_PATH}/train" \
    --total-images ${TRAIN_IMAGES}

python generate_dataset.py \
    --bgs-path ${BGS_PATH} \
    --templates-path  ${TEMPLATES_PATH}\
    --out-path "${OUT_PATH}/val" \
    --total-images ${VAL_IMAGES}

cd "${OUT_PATH}"
mkdir bigLabels
mkdir bigLabels/train
mv train/bigLabels/* bigLabels/train
mkdir bigLabels/val
mv val/bigLabels/* bigLabels/val
mkdir smallLabels
mkdir smallLabels/train
mv train/smallLabels/* smallLabels/train
mkdir smallLabels/val
mv val/smallLabels/* smallLabels/val
mkdir images
mkdir images/train
mv train/images/* images/train
mkdir images/val
mv val/images/* images/val

rm -rf train
rm -rf val