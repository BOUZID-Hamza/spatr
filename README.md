# SpATr : MoCap 3D Human Action Recognition based on Spiral Auto-encoder and Transformer Network

![model_overview2](https://github.com/h-bouzid/spatr/assets/94684114/b904166c-a979-4858-9444-b8dd7332d133)

## Abstract
Recent advancements in technology have expanded the possibilities of human action recognition by leveraging 3D data, which offers a richer representation of actions through the inclusion of depth information, enabling more accurate analysis of spatial and temporal characteristics. However, 3D human action recognition is a challenging task due to the irregularity and Disarrangement of the data points in action sequences. In this context, we present our novel model for human action recognition from fixed topology mesh sequences based on Spiral Auto-encoder and Transformer Network, namely SpATr. The proposed method first disentangles space and time in the mesh sequences. Then, an auto-encoder is utilized to extract spatial geometrical features, and a tiny transformer is used to capture the temporal evolution of the sequence. Previous methods either use 2D depth images, sample skeletons points, or they require a huge amount of memory leading to the ability to process short sequences only. In this work, we show a competitive recognition rate and high memory efficiency by building our auto-encoder based on spiral convolutions, which are lightweight convolutions directly applied to mesh data with fixed topologies, and by modeling temporal evolution using attention, that can handle large sequences. The proposed method is evaluated on two 3D human action datasets: MoVi and BMLrub from the Archive of Motion Capture As Surface Shapes (AMASS). The results analysis shows the effectiveness of our method in 3D human action recognition while maintaining high memory efficiency. The code will soon be made publicly available.

## Repository Requirements:
This code is compatible with PyTorch version 1.1, and we utilize tensorboardX for visualizing training metrics. We recommend setting up a virtual environment using Miniconda. To install PyTorch within a conda environment, execute the following command:

```
$ conda install pytorch torchvision -c pytorch
```
Subsequently, you can install the remaining prerequisites by running:
```
$ conda install pytorch torchvision -c pytorch
```
pip install -r requirements.txt

## Spiral Auto-Encoder

To extract spatial embeddings from the mesh data, we employed the "Neural 3D Morphable Models: Spiral Convolutional Networks for 3D Shape Representation Learning and Generation". For a more comprehensive understanding, please refer to the original work available on GitHub at: https://github.com/gbouritsas/Neural3DMM.

Please note that our research is built upon the utilization of the SMPL template and mesh decimation patterns during the training process. For your convenience, we have made available the template and decimation patterns within the "NeuralSMPL/dataset/template" directory. If you wish to make adjustments to the template, mesh decimation, filter sizes, or quantities, you can access the pertinent details at the following link: https://github.com/gbouritsas/Neural3DMM.

### data organization: 
https://github.com/gbouritsas/Neural3DMM#data-organization

### training
In this phase, we only use the training set.
We first split it into 80% for training the Spiral Auto-Encoder and 20% for validation. (we don't use the testing data until we want to perform the recognition).
```
$ python data_split.py
```
Then, in order to use a PyTorch dataloader for training and testing, we split the data into separate files by:
```
$ python data_generation.py --root_dir=/path/to/data_root_dir --dataset=DFAUST --num_valid=100
```
### testing
Then, we can use Jupyter Notebook to test the model:
```
$ jupyter notebook neural3dmm.ipynb
```
with 
```
$ args['mode'] = 'test'
```
## Data embedding extraction
```
$ args['mode'] = 'generate'
```
**NOTE: The extracted embedding of Babel, Movi, and BMLrub are available in LINK**

## Motion Transformer


