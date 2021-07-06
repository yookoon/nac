# Neural Activation Coding

This repository contains the code for the paper "Unsupervised Representation Learning via Neural Activation Coding" published on ICML 2021.

## Requirements
First install PyTorch. The code is tested for PyTorch 1.7.1.
- [PyTorch](https://pytorch.org)

Then run
```
pip install -r requirements.txt
```

## Training
To train the model on CIFAR-10
```
python run.py --objective=nac --optimizer=lars --lr=3.0 --lr_warmup=10 batch_size=1000 epochs=1000 --weight_decay=1e-6 --dropout=0.1
```
We used 8 TITAN X GPUs in our experiments.

### Linear evaluation using a model checkpoint
```
python linear.py --optimizer=sgd --lr=1.0 --batch_size=1000 --epochs=100 --weight_decay=0.0 --model_path=<path_to_your_checkpoint>
```

## Deep Hashing
TBD

## Acknowledgements
This repository is based on the [SimCLR implementation of leftthomas](https://github.com/leftthomas/SimCLR)
