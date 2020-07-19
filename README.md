# Federated Learning
Modification of [shaoxiongji](https://github.com/shaoxiongji/federated-learning)

## Requirements
python>=3.6  
pytorch>=0.4

## Run
Example:
> python main.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 10 --gpu 0  

NB: for CIFAR-10, `num_channels` must be 3.