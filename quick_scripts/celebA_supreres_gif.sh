python -u ./src/compressed_sensing.py \
    --pretrained-model-dir=./models/celebA_64_64/ \
    \
    --dataset celebA \
    --input-type full-input \
    --input-path-pattern "$1" \
    --num-input-images 1 \
    --batch-size 1 \
    \
    --measurement-type superres \
    --noise-std 0.0 \
    --superres-factor 4 \
    \
    --model-types dcgan \
    --mloss1_weight 0.0 \
    --mloss2_weight 1.0 \
    --zprior_weight 0.0 \
    --dloss1_weight 0.0 \
    --dloss2_weight 0.0 \
    \
    --optimizer-type adam \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --max-update-iter 30 \
    --num-random-restarts 1 \
    \
    --not-lazy \
    --print-stats \
    --checkpoint-iter 1 \
    --image-matrix 0 \
    --gif \
    --gif-iter 1 \
    --gif-dir ./gif/celebA/superres
