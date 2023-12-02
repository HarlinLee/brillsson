[Distilled Non-Semantic Speech Embeddings with Binary Neural Networks for Low-Resource Devices](https://www.sciencedirect.com/science/article/pii/S0167865523003380)
---
by Harlin Lee, Aaqib Saeed @ <a href="https://www.sciencedirect.com/journal/pattern-recognition-letters">Pattern Recognition Letters</a>.

### Abstract
This work introduces BRILLsson, a novel binary neural network-based representation learning model for a broad range of non-semantic speech tasks. We train the model with knowledge distillation from a large and real-valued TRILLsson model with only a fraction of the dataset used to train TRILLsson. The resulting BRILLsson models are only 2MB in size with a latency less than 8ms, making them suitable for deployment in low-resource devices such as wearables. We evaluate BRILLsson on eight benchmark tasks (including but not limited to spoken language identification, emotion recognition, health condition diagnosis, and keyword spotting), and demonstrate that our proposed ultra-light and low-latency models perform as well as large-scale models.

### Running the experiments
We provide 3 distilled (pre-trained) binary neural networks, and an example of linear probing with SpeechCommands dataset. 

#### 1. Installation
Install packages as follows:
```
pip3 install -r requirements.txt
```

#### 2. Distillation (Optional): 
```
python3 distill.py
```
Note that data loader has to be implemented.

#### 3. Run linear probing: 
```
python3 eval.py
```

### Citation
```
@article{lee2023distilled,
  title = {Distilled non-semantic speech embeddings with binary neural networks for low-resource devices},
  journal = {Pattern Recognition Letters},
  year = {2023},
  issn = {0167-8655},
  doi = {https://doi.org/10.1016/j.patrec.2023.11.028},
  url = {https://www.sciencedirect.com/science/article/pii/S0167865523003380},
  author = {Harlin Lee and Aaqib Saeed},
}
```
