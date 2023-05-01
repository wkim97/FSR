# Feature Separation and Recalibration for Adversarial Robustness (FSR)
Official Pytorch implementation of FSR.

[Feature Separation and Recalibration for Adversarial Robustness](https://arxiv.org/abs/2303.13846). \
[Woo Jae Kim](https://wkim97.github.io/), [Yoonki Cho](https://sgvr.kaist.ac.kr/member/), [Junsik Jung](https://sgvr.kaist.ac.kr/member/), [Sung-Eui Yoon](https://sgvr.kaist.ac.kr/~sungeui/) \
Korea Advanced Institute of Science and Technology (KAIST) \
in CVPR 2023 (Highlights Paper)

## Updates
[05/2023] Code is released.

## Feature Separation and Recalibration
![Framework of FSR](/figures/framework.png)
> (a) we propose a novel, easy-to-plugin approach named *Feature Separation and Recalibration (FSR)* that recalibrates the malicious, non-robust activations for more robust feature maps through Separation and Recalibration. 
(b) The Separation part disentangles the input feature map into the robust feature with activations that help the model make correct predictions and the non-robust feature with activations that are responsible for model mispredictions upon adversarial attack. 
(c) The Recalibration part then adjusts the non-robust activations to restore the potentially useful cues for model predictions.

## Setup
The packages necessary for running our code are provided in `environment.yml`. Create the conda environment `FSR` by running:
```
conda env create -f environment.yml
```
Note that if you're using more latest GPUs (e.g., RTX 3090), you may need to refer to [this pytorch link](https://pytorch.org/get-started/locally/) to install the PyTorch package that suits your cuda version.

## Running FSR
All codes for training and testing our FSR are provided in `run.sh`.

In this code, we only support running FSR with a single GPU, as our FSR module is light enough to be run on one GPU. 
While not implemented and tested, I expect that our code can also be run on multiple GPUs using `torch.nn.DataParallel()`.

### Training
The codes for training our FSR module can be found in `train.py`. 
Below is one example of training FSR on ResNet-18 using CIFAR-10 dataset:
```
python train.py \
--save_name cifar10_resnet18 --dataset cifar10 --model resnet18 --device 0
```
Please refer to `run.sh` for more training scripts and to `train.py` for more detailed arguments and their descriptions.

### Testing
After training, the model weights will be saved in `weights/[dataset]/[model]/[load_name].pth`. 
We load this checkpoint to evaluate the robustness of FSR module.
Below is one example of testing FSR on ResNet-18 using CIFAR-10 dataset:
```
python test.py \
--load_name cifar10_resnet18 --dataset cifar10 --model resnet18 --device 0
```
Please refer to `run.sh` for more testing scripts and to `test.py` for more detailed arguments and their descriptions.

## Citation
If you find our work useful, please consider using the following citation:
```
@inproceedings{kim2023feature,
  title={Feature Separation and Recalibration for Adversarial Robustness},
  author={Kim, Woo Jae and Cho, Yoonki and Jung, Junsik and Yoon, Sung-Eui},
  booktitle={CVPR},
  year={2023}
}
```

## Acknowledgement
We thank the authors of [CAS](https://github.com/bymavis/CAS_ICLR2021) and [CIFS](https://github.com/HanshuYAN/CIFS) for their contribution to the field and our research. 
Our implementation is inspired by or utilized parts of CAS and CIFS as credited in our code.

This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (No. RS-2023-00208506).