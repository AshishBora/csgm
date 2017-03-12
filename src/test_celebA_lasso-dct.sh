python -u compressed_sensing.py \
    --dataset celebA \
    --input-type full-input \
    --num-input-images 10 \
    --batch-size 10 \
    \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 50 \
    \
    --model-types lasso-dct \
    --lmbd 0.1 \
    \
    --not-lazy \
    --print-stats \
    --checkpoint-iter 1 \
    --image-matrix 1
