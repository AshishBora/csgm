python -u compressed_sensing.py \
    --dataset mnist \
    --input-type full-input \
    --num-input-images 10 \
    --batch-size 10 \
    \
    --measurement-type learned \
    --noise-std 0.0 \
    --num-measurements 30 \
    \
    --model-types learned \
    \
    --not-lazy \
    --print-stats \
    --checkpoint-iter 1 \
    --image-matrix 1
