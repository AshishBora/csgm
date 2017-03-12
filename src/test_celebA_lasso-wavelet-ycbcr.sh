python -u compressed_sensing.py \
    --dataset celebA \
    --input-type full-input \
    --num-input-images 1 \
    --batch-size 1 \
    \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 2500 \
    \
    --model-types lasso-wavelet-ycbcr \
    --lasso-solver cvxopt \
    --lmbd 0.00001 \
    \
    --not-lazy \
    --print-stats \
    --checkpoint-iter 1 \
    --image-matrix 1
