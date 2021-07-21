# Neural Activation Coding

This repository contains the code for the paper "Unsupervised Representation Learning via Neural Activation Coding" published on ICML 2021.

## Requirements
First install PyTorch. The code is tested on PyTorch 1.7.1.
- [PyTorch](https://pytorch.org)

Then run
```
pip install -r requirements.txt
```

## Linear Classification
To train a ResNet-50 model on CIFAR-10
```
python run.py --objective=nac --optimizer=lars --lr=3.0 --lr_warmup=10 batch_size=1000 epochs=1000 --weight_decay=1e-6 --flip=0.1
```
We used 4 TITAN RTX GPUs in our experiments.

## Deep Hashing
To train a VGG-16 model on the subset of CIFAR-10
```
python run_hash.py --objective=nac --optimizer=lars --lr=3.0 --lr_warmup=100 batch_size=1000 epochs=2000 --weight_decay=1e-6 --flip=0.4
```

## Acknowledgements
This repository is based on the [SimCLR implementation of leftthomas](https://github.com/leftthomas/SimCLR)
