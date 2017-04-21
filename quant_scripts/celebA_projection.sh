# BASE_SCRIPT = [
#     "python -u ./src/compressed_sensing.py \\",
#     "    --pretrained-model-dir=./models/celebA_64_64/ \\",
#     "    \\",
#     "    --dataset celebA \\",
#     "    --input-type full-input \\",
#     "    --num-input-images 64 \\",
#     "    --batch-size 64 \\",
#     "    \\",
#     "    --measurement-type project \\",
#     "    --noise-std 0.01 \\",
#     "    --num-measurements 50 \\",
#     "    \\",
#     "    --model-types k-sparse-wavelet \\",
#     "    --mloss1_weight 0.0 \\",
#     "    --mloss2_weight 1.0 \\",
#     "    --zprior_weight 0.001 \\",
#     "    --dloss1_weight 0.0 \\",
#     "    --dloss2_weight 0.0 \\",
#     "    --lmbd 0.1 \\",
#     "    --sparsity 50 \\",
#     "    \\",
#     "    --optimizer-type adam \\",
#     "    --learning-rate 0.1 \\",
#     "    --momentum 0.9 \\",
#     "    --max-update-iter 500 \\",
#     "    --num-random-restarts 2 \\",
#     "    \\",
#     "    --save-images \\",
#     "    --save-stats \\",
#     "    --print-stats \\",
#     "    --checkpoint-iter 1 \\",
#     "    --image-matrix 0",
# ]

cd src

# k-sparse (Wavelet)
python create_scripts.py \
    --input-type full-input \
    --measurement-type project \
    --noise-std 0.0 \
    --num-measurements 12288 \
    --model-types k-sparse-wavelet \
    --lmbd 0.1 \
    --sparsity 50 100 200 500 1000 2500 5000 7500 10000 12288
