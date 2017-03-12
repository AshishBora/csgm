cd src
python -u compressed_sensing.py \
    --dataset=mnist \
    --input-type=full-input \
    --num-input-images=100 \
    --batch-size=50 \
    \
    --measurement-type inpaint \
    --noise-std=0.0 \
    --inpaint-size=0 \
    \
    --model-types vae \
    --mloss1_weight 0.0 \
    --mloss2_weight 1.0 \
    --zprior_weight 0.0 \
    --lmbd=0.1 \
    \
    --optimizer-type=adam \
    --learning-rate=0.1 \
    --momentum=0.9 \
    --max-update-iter=1000 \
    --num-random-restarts=10 \
    \
    --save-images \
    --save-stats \
    --print-stats \
    --checkpoint-iter=1 \
    --image-matrix=0
