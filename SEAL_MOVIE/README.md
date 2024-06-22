SEAL\_MOVIE -- An Implementation of SEAL for Movie Co-star Link Prediction Task
===============================================================================

About
-----

SEAL is a GNN-based link prediction method. It first extracts a k-hop enclosing subgraph for each target link, then applies a labeling trick named Double Radius Node Labeling (DRNL) to give each node an integer label as its additional feature. Finally, these labeled enclosing subgraphs are fed to a graph neural network to predict link existences.


SEAL_MOVIE in this repository is a customized version from SEAL_OGB. The original implementation is [here](https://github.com/facebookresearch/SEAL_OGB).

The original paper of SEAL is:
> M. Zhang and Y. Chen, Link Prediction Based on Graph Neural Networks, Advances in Neural Information Processing Systems (NIPS-18). [\[PDF\]](https://arxiv.org/pdf/1802.09691.pdf)

This repository also implements some other labeling tricks, such as Distance Encoding (DE) and Zero-One (ZO), and supports combining labeling tricks with different GNNs, including GCN, GraphSAGE and GIN.

Requirements
------------

Latest tested combination: Python 3.8.5 + PyTorch 1.6.0 + PyTorch\_Geometric 1.6.1

Install [PyTorch](https://pytorch.org/)

Install [PyTorch\_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Other required python libraries include: numpy, scipy, tqdm etc.

Usages
------

### movie-actor

    python seal_link_pred.py --dataset movie_actor --num_hops 1 --num_layers 3 --use_feature --custom_split --eval_steps 5 --hidden_channels 256 --batch_size 32 --epochs 50 --runs 5 --model GCN 

License
-------

SEAL\_OGB is released under an MIT license. Find out more about it [here](https://github.com/facebookresearch/SEAL_OGB/blob/master/LICENSE).

Reference
---------

Citation of SEAL_OGB and Facebook AI Team, really appreciate it.

	@article{zhang2021labeling,
      title={Labeling Trick: A Theory of Using Graph Neural Networks for Multi-Node Representation Learning},
      author={Zhang, Muhan and Li, Pan and Xia, Yinglong and Wang, Kai and Jin, Long},
      journal={Advances in Neural Information Processing Systems},
      volume={34},
      year={2021}
    }

    @inproceedings{zhang2018link,
      title={Link prediction based on graph neural networks},
      author={Zhang, Muhan and Chen, Yixin},
      booktitle={Advances in Neural Information Processing Systems},
      pages={5165--5175},
      year={2018}
    }

Muhan Zhang, Facebook AI

10/13/2020
