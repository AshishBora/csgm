cd src

# DCGAN
python -u compressed_sensing.py \
    --dataset=celebA \
    --input-type=full-input \
    --num-input-images=64 \
    --batch-size=64 \
    \
    --measurement-type superres \
    --noise-std=0.0 \
    --superres-factor=4 \
    \
    --model-types dcgan \
    --mloss1_weight 0.0 \
    --mloss2_weight 1.0 \
    --zprior_weight 0.0 \
    --dloss1_weight 0.0 \
    --dloss2_weight 0.0 \
    \
    --optimizer-type=adam \
    --learning-rate=0.1 \
    --momentum=0.9 \
    --max-update-iter=1000 \
    --num-random-restarts=2 \
    \
    --save-images \
    --save-stats \
    --print-stats \
    --checkpoint-iter=1 \
    --image-matrix=0


# DCGAN + D
python -u compressed_sensing.py \
    --dataset=celebA \
    --input-type=full-input \
    --num-input-images=64 \
    --batch-size=64 \
    \
    --measurement-type superres \
    --noise-std=0.0 \
    --superres-factor=4 \
    \
    --model-types dcgan \
    --mloss1_weight 0.0 \
    --mloss2_weight 1.0 \
    --zprior_weight 0.0 \
    --dloss1_weight 1.0 \
    --dloss2_weight 0.0 \
    \
    --optimizer-type=adam \
    --learning-rate=0.1 \
    --momentum=0.9 \
    --max-update-iter=1000 \
    --num-random-restarts=2 \
    \
    --save-images \
    --save-stats \
    --print-stats \
    --checkpoint-iter=1 \
    --image-matrix=0
