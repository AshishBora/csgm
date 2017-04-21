python -u ./src/compressed_sensing.py \
    --pretrained-model-dir=./mnist-vae/models/mnist-vae/ \
    \
    --dataset mnist \
    --input-type full-input \
    --num-input-images 10 \
    --batch-size 10 \
    \
    --measurement-type project \
    --noise-std 0.0 \
    \
    --model-types vae \
    --mloss1_weight 0.0 \
    --mloss2_weight 1.0 \
    --zprior_weight 0.0 \
    --lmbd 0.1 \
    \
    --optimizer-type adam \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --max-update-iter 1000 \
    --num-random-restarts 10 \
    \
    --not-lazy  \
    --print-stats \
    --checkpoint-iter 1 \
    --image-matrix 1
