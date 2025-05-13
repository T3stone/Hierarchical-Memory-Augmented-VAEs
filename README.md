
# Hierarchical Memory-Augmented Variational Autoencoders: Enhancing Robust Image Generation with Multi-Scale Feature Retention

This repository is the official implementation of the paper **"Hierarchical Memory-Augmented Variational Autoencoders: Enhancing Robust Image Generation with Multi-Scale Feature Retention"** (manuscript in active submission to *The Visual Computer*). Our work highlights the potential of integrating brain-inspired hierarchical memory structures with probabilistic deep generative models to advance image synthesis capabilities through multi-scale feature retention.


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
To set up the env environment
```setup
conda create -n myenv python=3.10
```
```setup
conda install numpy=1.24.3 matplotlib=3.8.4 scikit-learn=1.4.2 
```
```setup
conda install pytorch=2.3.1 torchvision=0.18.1 torchaudio=2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```
```setup
pip install pytorch-fid==0.3.0 pytorch-msssim==1.0.0
```

## Evaluation

To evaluate my model on MNIST or Fasion-MNIST, run:

```FID
python test-fid.ipynb  --benchmark MNIST or Fasion-MNIST
```
```PRDC
python PRDC-test.ipynb --benchmark MNIST or Fasion-MNIST
```
```Different Noises
python test-noise.ipynb --benchmark MNIST or Fasion-MNIST
```



