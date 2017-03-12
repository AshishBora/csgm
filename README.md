# Compressed Sensing using Generative Models

This repository provides code to reproduce results from the paper: [Compressed Sensing using Generative Models](https://arxiv.org/abs/1703.03208)

### Steps to reproduce the results:
---

#### Preliminaries
---

Download the datasets
```shell
$ python download.py mnist
$ python download.py celebA
```

Make a new folder ```data/celebA_test``` and put some images in that folder. These images will be used for testing the algorithms.

Download pretrained models from <https://drive.google.com/open?id=0B0ox77cWXKmLLUhtWEt3RGdpczg> and put them in ```models/```

To use wavelet based estimators, you need to run ```$ python ./src/wavelet_basis.py``` to create the wavelet basis matrix.


#### Experiments
---
These are the supported experiments:

1. Reconstruction from Gaussian measurements
2. Super-resolution
3. Sensing Images from the span of the generator (gen-span)
4. Quantifying representation error (projection)

To quickly see the reconstructions, use the ```src/test_{dataset}_{model_type}.sh``` files.

To reproduce the quantitative results, you have to run a lot of scripts. To make the job easier, we have a script generator.

1. Identfy an experiment (`expt`) that you'd like to reproduce, and locate the file ```{dataset}_{expt}.sh```.

2. Then make sure the ```BASE_SCRIPTS``` in ```src/create_scripts.py``` is same as the one at the top of ```{dataset}_{expt}.sh```.

3. Run  ```$ ./{dataset}_{expt}.sh```. This will create a bunch of ```.sh``` files in the ```scripts/``` directory, each one of them for a different parameter setting.

4. To start running these scripts, you can run ```$ ./utils/run_sequentially.sh``` to run them one by one. Alternatively use ```$ ./utils/run_all_by_number.sh``` to create screens and start proccessing them in parallel. [WARNING: This may overwhelm the computer.] You can use ```$ ./utils/stop_all_by_number.sh``` to stop the running processes started this way. 

This will run the experiments with all the parameter settings and save the results to appropriately named directories. Once this is done, to get the plots, head over to ```src/metrics.ipynb```. To view reconstructed images, you can either find them in ```estimated/``` or to get the matrix of images (as in the paper), see ```src/view_est_{dataset}.ipynb```.
