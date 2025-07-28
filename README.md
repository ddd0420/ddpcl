# ddpcl
The official implementation of "Dual diversity and pseudo-label correction learning for semi-supervised medical image segmentation"

## Requirements
This repository is based on PyTorch 2.1.1, CUDA 12.2 and Python 3.9.21. All experiments in our paper were conducted on NVIDIA GeForce RTX 4090 GPU with an identical experimental setting.


To train a model,
```
python train_ACDC.py  #for ACDC training
``` 
To test a model,
```
python test_ACDC.py  #for ACDC testing
```


## Acknowledgements
Our code is largely based on [BCP](https://github.com/DeepMed-Lab-ECNU/BCP) and SSL4MIS(https://github.com/HiLab-git/SSL4MIS)). Thanks for these authors for their valuable work, hope our work can also contribute to related research.
