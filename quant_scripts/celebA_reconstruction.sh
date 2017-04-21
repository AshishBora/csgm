# BASE_SCRIPT = [
#     "python -u ./src/compressed_sensing.py \\",
#     "    --pretrained-model-dir=./models/celebA_64_64/ \\",
#     "    \\",
#     "    --dataset celebA \\",
#     "    --input-type full-input \\",
#     "    --num-input-images 64 \\",
#     "    --batch-size 64 \\",
#     "    \\",
#     "    --measurement-type gaussian \\",
#     "    --noise-std 0.01 \\",
#     "    --num-measurements 50 \\",
#     "    \\",
#     "    --model-types dcgan \\",
#     "    --mloss1_weight 0.0 \\",
#     "    --mloss2_weight 1.0 \\",
#     "    --zprior_weight 0.001 \\",
#     "    --dloss1_weight 0.0 \\",
#     "    --dloss2_weight 0.0 \\",
#     "    --lmbd 0.1 \\",
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

# Lasso (DCT - RGB)
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 20 50 100 200 500 1000 2500 5000 7500 10000 \
    --model-types lasso-dct \
    --lmbd 0.1

# Lasso (Wavelet - RGB)
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 20 50 100 200 500 1000 2500 5000 7500 10000 \
    --model-types lasso-wavelet \
    --lmbd 0.00001

# DCGAN
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 20 50 100 200 500 1000 2500 5000 7500 10000 \
    --model-types dcgan \
    --zprior_weight 0.0 \
    --dloss1_weight 0.0 \
    --max-update-iter 500 \
    --num-random-restarts 2

# DCGAN + Reg
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 20 50 100 200 500 1000 2500 5000 7500 10000 \
    --model-types dcgan \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0 1.0 \
    --max-update-iter 500 \
    --num-random-restarts 10

# DCGAN + Reg : From generator
python create_scripts.py \
    --input-type gen-span \
    --measurement-type gaussian \
    --noise-std 0.01 \
    --num-measurements 20 50 100 200 500 1000 2500 \
    --model-types dcgan \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0 \
    --max-update-iter 500 \
    --num-random-restarts 1

# DCGAN + Reg : Err vs noise
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 1.0 10.0 20.0 50.0 100.0 \
    --num-measurements 2500 \
    --model-types dcgan \
    --zprior_weight 0.001 \
    --dloss1_weight 0.0 \
    --max-update-iter 5000 \
    --num-random-restarts 1

# Lasso (Wavelet) : Err vs Noise
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 1.0 10.0 20.0 50.0 100.0 \
    --num-measurements 2500 \
    --model-types lasso-wavelet \
    --lmbd 0.1

##### Lasso (DCT) : Err vs Noise
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 1.0 10.0 20.0 50.0 100.0 \
    --num-measurements 2500 \
    --model-types lasso-dct \
    --lmbd 0.1
