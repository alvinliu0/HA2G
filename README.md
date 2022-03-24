# Learning Hierarchical Cross-Modal Association for Co-Speech Gesture Generation (CVPR 2022)

Xian Liu, Qianyi Wu, Hang Zhou, Yinghao Xu, Rui Qian, Xinyi Lin, Xiaowei Zhou, Wayne Wu, Bo Dai, and Bolei Zhou.

### [Project](https://alvinliu0.github.io/projects/HA2G) | [Paper](https://arxiv.org/abs/xxxx) | [Demo](https://www.youtube.com/watch?v=CG632W-nIWk)

Generating speech-consistent body and gesture movements is a long-standing problem in virtual avatar creation. Previous studies often synthesize pose movement in a holistic manner, where poses of all joints are generated simultaneously. Such a straightforward pipeline fails to generate fine-grained co-speech gestures. One observation is that the hierarchical semantics in speech and the hierarchical structures of human gestures can be naturally described into multiple granularities and associated together. To fully utilize the rich connections between speech audio and human gestures, we propose a novel framework named **Hierarchical Audio-to-Gesture (HA2G)** for co-speech gesture generation. In HA2G, a Hierarchical Audio Learner extracts audio representations across semantic granularities. A Hierarchical Pose Inferer subsequently renders the entire human pose gradually in a hierarchical manner. To enhance the quality of synthesized gestures, we develop a contrastive learning strategy based on audio-text alignment for better audio representations. Extensive experiments and human evaluation demonstrate that the proposed method renders realistic co-speech gestures and outperforms previous methods in a clear margin.

<img src='./misc/HA2G.png' width=800>

## Code

The code is coming soon. Stay tuned!

## Citation

If you find our work useful, please kindly cite as:
```
@inproceedings{liu2022learning,
title={Learning Hierarchical Cross-Modal Association for Co-Speech Gesture Generation},
author={Liu, Xian and Wu, Qianyi and Zhou, Hang and Xu, Yinghao and Qian, Rui and Lin, Xinyi and Zhou, Xiaowei and Wu, Wayne and Dai, Bo and Zhou, Bolei},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2022}
}
```
