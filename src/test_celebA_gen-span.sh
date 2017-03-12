python -u compressed_sensing.py \
    --dataset celebA \
    --input-type gen-span \
    --num-input-images 64 \
    --batch-size 64 \
    \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 50 \
    \
    --model-types dcgan \
    --mloss1_weight 0.0 \
    --mloss2_weight 1.0 \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0 \
    --dloss2_weight 0.0 \
    \
    --optimizer-type adam \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --max-update-iter 2 \
    --num-random-restarts 2 \
    \
    --not-lazy \
    --print-stats \
    --checkpoint-iter 1 \
    --image-matrix 1
