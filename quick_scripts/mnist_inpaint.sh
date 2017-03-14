python -u ./src/compressed_sensing.py \
    --dataset mnist \
    --input-type full-input \
    --num-input-images 10 \
    --batch-size 10 \
    \
    --measurement-type inpaint \
    --noise-std 0.0 \
    --inpaint-size 10 \
    \
    --model-types vae \
    --mloss1_weight 0.0 \
    --mloss2_weight 1.0 \
    --zprior_weight 0.0 \
    --lmbd 0.1 \
    \
    --optimizer-type momentum \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --max-update-iter 100 \
    --num-random-restarts 10 \
    \
    --not-lazy  \
    --print-stats \
    --checkpoint-iter 1 \
    --image-matrix 1
