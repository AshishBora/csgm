python -u ./src/compressed_sensing.py \
    --pretrained-model-dir=./models/celebA_64_64/ \
    \
    --dataset celebA \
    --input-type full-input \
    --input-path-pattern "$1" \
    --num-input-images 1 \
    --batch-size 1 \
    \
    --measurement-type project \
    --noise-std 0.0 \
    \
    --model-types k-sparse-wavelet \
    --sparsity 5000 \
    \
    --not-lazy \
    --print-stats \
    --checkpoint-iter 1 \
    --image-matrix 1
