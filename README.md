# Face Forgery Detection by 3D Decomposition and Composition Search
## Demo
<img src="./whole_graph_small.gif" alt="whole_graph_small" style="zoom: 33%;" />

## Introduction

With the development of deep learning, deep learning models have become far more capable than humans in areas such as digital forgery detection. Many forgery samples that cannot be detected by the human eye can be accurately inferred and classified by the models. It is natural to think about extracting the visual clues and inferences used by deep learning models to guide the identification of unknown forgeries. However, the black-box characteristics of deep learning models lead to the fact that this knowledge is encoded in the weights of the neural network that cannot be parsed, which limits the possibility of knowledge transfer. The interpretable knowledge in the attention graph obtained from the visualization operation is very limited.

Our approach is motivated by the hope that the network discovers knowledge during training, and that this knowledge can be extracted to guide humans in designing neural networks. Guided by this idea, we designed a more general clue-searching method that can be used to search for optimal paths and actively discover the visual clues and levels of inference, as long as a given task can be decomposed into multiple nodes and the nodes can be combined. We have first applied the method to the task of face forgery detection with good results, and will subsequently support other related tasks.

Specifically, we use inference graphs to facilitate the extraction of this important knowledge. Since the nodes and edges of the inference graph are interpretable, the optimal path can be used directly as knowledge to reveal the visual clues and levels of inference learned by models. We applied this method to face forgery detection and successfully found that facial personality texture and ambient direct light contain the most forgery clues.

## Citation
If you find this project useful in your research, please consider cite:
```
@article{zhu2023face,
  title={Face Forgery Detection by 3D Decomposition and Composition Search},
  author={Zhu, Xiangyu and Fei, Hongyan and Zhang, Bin and Zhang, Tianshuo and Zhang, Xiaoyu and Li, Stan Z and Lei, Zhen},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```
