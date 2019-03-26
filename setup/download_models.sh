mkdir models
mkdir optimization
mkdir mnist_vae/models

unzip csgm_pretrained.zip
mv csgm_pretrained/celebA_64_64/ models/
mv csgm_pretrained/mnist-e2e/ optimization/
mv csgm_pretrained/mnist-vae/ mnist_vae/models/

rm -r csgm_pretrained
