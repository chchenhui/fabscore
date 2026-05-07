# IVTFuse: An Efficient Vision-Language Guided Infrared-Visible Image Fusion Network with Frequency-Strip and Hybrid Pooling Attention Modules

This repository is the official implementation of the **IVTFuse** network.  
The pretrained checkpoint can be found at:

```bash
models/ckpt_100.pth
```


## 1. Features

- Tri-modal fusion: Infrared, Visible, and BLIP-encoded text.

- Lightweight attention modules: FSA (frequency-selective) and HPA (spatial pooling).



## 2. Training:

Please follow the FILM repository at https://github.com/Zhaozixiang1228/IF-FILM/tree/main for the Environment setup, Data preparation and Pre-processing steps.


Once your environment and data are ready, train IVTFuse with:

```bash
python train.py 

```


### 3. Module-level Ablations:

We expose two flags in Net:

- use_fsa (default: True)

- use_hpa (default: True)

Example runs:

```bash

# w/o FSA and HPA
python train.py ... --use_fsa 0 --use_hpa 0

# w/o FSA
python train.py ... --use_fsa 0 --use_hpa 1

# w/o HPA
python train.py ... --use_fsa 1 --use_hpa 0

# Full
python train.py ... --use_fsa 1 --use_hpa 1

```

## 2. Testing:

Test the IVTFuse with:

```bash
python test.py 

```


