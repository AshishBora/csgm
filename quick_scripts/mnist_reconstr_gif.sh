python -u ./src/compressed_sensing.py \
    --pretrained-model-dir=./mnist_vae/models/mnist-vae/ \
    \
    --dataset mnist \
    --input-type full-input \
    --num-input-images 20 \
    --batch-size 20 \
    \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 100 \
    \
    --model-types lasso vae \
    --mloss1_weight 0.0 \
    --mloss2_weight 1.0 \
    --zprior_weight 0.1 \
    --lmbd 0.1 \
    \
    --optimizer-type adam \
    --learning-rate 0.01 \
    --momentum 0.9 \
    --max-update-iter 1000 \
    --num-random-restarts 1 \
    \
    --not-lazy \
    --print-stats \
    --checkpoint-iter 1 \
    --image-matrix 0 \
    --gif \
    --gif-iter 25 \
    --gif-dir ./gif/mnist/reconstr
