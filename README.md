
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
## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on MNIST or Fasion-MNIST, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet]

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
