# BASE_SCRIPT   [
#     "python -u ./src/compressed_sensing.py \\",
#     "    --dataset mnist \\",
#     "    --input-type full-input \\",
#     "    --num-input-images 300 \\",
#     "    --batch-size 50 \\",
#     "    \\",
#     "    --measurement-type gaussian \\",
#     "    --noise-std 0.1 \\",
#     "    --num-measurements 10 \\",
#     "    \\",
#     "    --model-types vae \\",
#     "    --mloss1_weight 0.0 \\",
#     "    --mloss2_weight 1.0 \\",
#     "    --zprior_weight 0.1 \\",
#     "    --dloss1_weight 0.0 \\",
#     "    --lmbd 0.1 \\",
#     "    \\",
#     "    --optimizer-type adam \\",
#     "    --learning-rate 0.01 \\",
#     "    --momentum 0.9 \\",
#     "    --max-update-iter 1000 \\",
#     "    --num-random-restarts 10 \\",
#     "    \\",
#     "    --save-images \\",
#     "    --save-stats \\",
#     "    --print-stats \\",
#     "    --checkpoint-iter 1 \\",
#     "    --image-matrix 0",
#     "",
# ]

cd src

# Lasso
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 10 25 50 100 200 300 400 500 750 \
    --model-types lasso \
    --lmbd 0.1

# VAE and VAE+Reg
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 10 25 50 100 200 300 400 500 750 \
    --model-types vae \
    --zprior_weight 0.0 0.1 \
    --max-update-iter 1000 \
    --num-random-restarts 10

# VAE: From generator
python create_scripts.py \
    --input-type gen-span \
    --measurement-type gaussian \
    --noise-std 0.1 \
    --num-measurements 10 25 50 100 200 300 400 500 750 \
    --model-types vae \
    --zprior_weight 0.0 0.1 \
    --max-update-iter 1000 \
    --num-random-restarts 10

# E2E
python create_scripts.py \
    --input-type full-input \
    --measurement-type fixed learned \
    --noise-std 0.1 \
    --num-measurements 10 20 30 \
    --model-types learned

# Lasso : Error vs noise
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 0.1 1.0 2.0 5.0 10.0 20.0 50.0 \
    --model-types lasso \
    --num-measurements 500 \
    --lmbd 0.1

# VAE : Error vs noise
python create_scripts.py \
    --input-type full-input \
    --measurement-type gaussian \
    --noise-std 0.01 0.1 1.0 2.0 5.0 10.0 20.0 50.0 \
    --model-types vae \
    --num-measurements 100 500 \
    --zprior_weight 0.1 \
    --max-update-iter 1000 \
    --num-random-restarts 2
