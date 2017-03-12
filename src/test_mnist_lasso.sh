python -u compressed_sensing.py \
    --dataset mnist \
    --input-type full-input \
    --num-input-images 10 \
    --batch-size 10 \
    \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 400 \
    \
    --model-types lasso \
    --lmbd 0.1 \
    \
    --not-lazy \
    --print-stats \
    --checkpoint-iter 1 \
    --image-matrix 1
