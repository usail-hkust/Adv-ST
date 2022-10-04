# Practical Adversarial Attacks on Spatiotemporal Traffic Forecasting Models



> Machine learning based traffic forecasting models leverage sophisticated spatiotemporal auto-correlations to provide accurate predictions of city-wide traffic states. 
However, existing methods assume a reliable and unbiased forecasting environment, which is not always available in the wild.
In this work, we investigate the vulnerability of spatiotemporal traffic forecasting models and propose a practical adversarial spatiotemporal attack framework.
Specifically, instead of simultaneously attacking all geo-distributed data sources, an iterative gradient-guided node saliency method is proposed to identify the time-dependent set of victim nodes.
Furthermore, we devise a spatiotemporal gradient descent based scheme to generate real-valued adversarial traffic states under a perturbation constraint.
Meanwhile, we theoretically demonstrate the worst performance bound of adversarial traffic forecasting attacks.
Extensive experiments on two real-world datasets show that the proposed two-step framework achieves up to 67.8% performance degradation on various advanced spatiotemporal forecasting models.
Remarkably, we also show that adversarial training with our proposed attacks can significantly improve the robustness of spatiotemporal traffic forecasting models.

This repository includes:
- Code for the ASTFA in our study.
- Code and other baselines for our method.

A pytorch implementation for the paper:

Practical Adversarial Attacks on Spatiotemporal
Traffic Forecasting Models in NeurIPS 2022

## Environment 
* [PyTorch](https://pytorch.org/) (tested on 1.8.0)
* [mmcv](https://github.com/open-mmlab/mmcv)


## Datasets
We use the METR-LA and PeMS-Bay datasets ([link](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX)). 

## Usage
### Adversarial spatiotemporal traffic forecasting attack framework and Baseline
To train and evaluate a baseline model, run the following commands:
```
# white-box attack  METR-LA
python train.py configs/METRLA/METRLA-train0.7-val0.1-test0.2-in12out12-0.1nodes-gwnet-standard.yaml
python test.py configs/METRLA/METRLA-train0.7-val0.1-test0.2-in12out12-0.1nodes-gwnet-standard.yaml -a ALL

# white-box attack  PeMS-Bay
python train.py configs/PeMS/PeMS-train0.7-val0.1-test0.2-in12out12-0.1nodes-gwnet-standard.yaml
python test.py configs/PeMS/PeMS-train0.7-val0.1-test0.2-in12out12-0.1nodes-gwnet-standard.yaml -a ALL

# Grey-box attack  METR-LA
python grey_rain.py configs/SemiMETRLA/SemiMETRLA-train0.7-val0.1-test0.2-in12out12-0.1nodes-stawnet-standard.yaml
python grey_test.py configs/SemiMETRLA/SemiMETRLA-train0.7-val0.1-test0.2-in12out12-0.1nodes-stawnet-standard.yaml -a grey-attack

# Grey-box attack PeMS-Bay
python grey_train.py configs/SemiPeMS/PeMS-train0.7-val0.1-test0.2-in12out12-0.1nodes-gwnet-standard.yaml
python grey_test.py configs/SemiPeMS/PeMS-train0.7-val0.1-test0.2-in12out12-0.1nodes-gwnet-standard.yaml -a grey-attack

# Black-box attack  METR-LA
python grey_test.py configs/BlackMETRLA/SemiMETRLA-train0.7-val0.1-test0.2-in12out12-0.1nodes-stawnet-standard.yaml -a black-attack

# Black-box attack PeMS-Bay
python grey_test.py configs/BlackPeMS/BlackMETRLA-train0.7-val0.1-test0.2-numsteps12-0.1nodes-source_stawnet-target_astgcn.yaml -a black-attack

```
Here `-a ALL` ,`-a grey-attack`, and `-a black-attack` denote that we evaluate attacks including STPGD-TNDS, STMIM-TNDS, PGD-Random, PGD-PR, PGD-Centrality, PGD-Degree, MIM-Random, MIM-PR, MIM-Centrality, MIM-Degree .


## License and Citation
If you find our code or paper useful, please cite our paper:
```bibtex
@inproceedings{fan2022ASTFA,
 author =  {Fan LIU, Hao LIU, Wenzhao JIANG},
 title = {Practical Adversarial Attacks on Spatiotemporal
Traffic Forecasting Models},
 booktitle = {In Proceedings of the Thirty-sixth Annual Conference on Neural Information Processing Systems (NeurIPS)},
 year = {2022}
 }
```
## Acknowledgement
We thank the authors for the following repositories for code reference:
[Adversarial Long-Tail](https://github.com/wutong16/Adversarial_Long-Tail), etc.

## Contact
Please contact [@fan](https://github.com/luckyfan-cs) for questions, comments and reporting bugs.
