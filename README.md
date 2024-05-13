# SLAB

### **SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized BatchNorm**

*Jialong Guo\*, Xinghao Chen\*, Yehui Tang, Yunhe Wang (\*Equal Contribution)*

*arXiv 2024* 

[[`arXiv`](https://arxiv.org/abs/xx)] [[`BibTeX`](#citation)]

## 🔥 Updates
* **2024/05/13**: Pre-trained models and codes of SLAB are released both in [Pytorch](https://github.com/xinghaochen/SLAB) and [Mindspore](https://github.com/mindspore-lab/models/tree/master/research/huawei-noah/SLAB).

## 📸 Overview
This is an official mindspore implementation of our paper "**SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized BatchNorm**". In this paper, we investigate the computational bottleneck modules of efficient transformer, i.e., normalization layers and attention modules. Layer normalization is commonly used in transformer architectures but is not computational friendly due to statistic calculation during inference. However, replacing Layernorm with more efficient batch normalization in transformer often leads to inferior performance and collapse in training. To address this problem, we propose a novel method named PRepBN to progressively replace LayerNorm with re-parameterized BatchNorm in training. During inference, the proposed PRepBN could be simply re-parameterized into a normal BatchNorm, thus could be fused with linear layers to reduce the latency. Moreover, we propose a simplified linear attention (SLA) module that is simply yet effective to achieve strong performance. Extensive experiments on image classification as well as object detection demonstrate the effectiveness of our proposed method. For example, powered by the proposed methods, our SLAB-Swin obtains 
83.6% top-1 accuracy on ImageNet with 16.2ms latency, which is 2.4ms less than that of Flatten-Swin with 0.1 higher accuracy.


## ✏️ Reference
If you find SqueezeTime useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:
```
@article{guo2024slab,
  title={SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized BatchNorm},
  author={Guo, Jialong and Chen, Xinghao and Tang, Yehui  and Wang, Yunhe},
  journal={arXiv preprint},
  year={2024}
}
```
