# Understanding and Improving Early Stopping for Learning with Noisy Labels

PyTorch Code for the following paper at NeurIPS 2021:\
<b>Title</b>: Understanding and Improving Early Stopping for Learning with Noisy Labels \
<b>Authors</b>: Yingbin Bai*, Erkun Yang*, Bo Han, Yanhua Yang, Jiatong Li, Yinian Mao, Gang Niu, and Tongliang Liu


## Experiments

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋 Please download and place all datasets into the data directory. For Clohting1M, please run "python ClothingData.npy" to generate a data file.

To train PES without semi on CIFAR-10/100

```
python PES_cs.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5
```