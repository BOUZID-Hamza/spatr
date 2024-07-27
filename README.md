# SpATr : MoCap 3D Human Action Recognition based on Spiral Auto-encoder and Transformer Network

![model_overview2](https://github.com/h-bouzid/spatr/assets/94684114/b904166c-a979-4858-9444-b8dd7332d133)

## Abstract
Recent advancements in technology have expanded the possibilities of human action recognition by leveraging 3D data, which offers a richer representation of actions through the inclusion of depth information, enabling more accurate analysis of spatial and temporal characteristics. However, 3D human action recognition is a challenging task due to the irregularity and Disarrangement of the data points in action sequences. In this context, we present our novel model for human action recognition from fixed topology mesh sequences based on Spiral Auto-encoder and Transformer Network, namely SpATr. The proposed method first disentangles space and time in the mesh sequences. Then, an auto-encoder is utilized to extract spatial geometrical features, and a tiny transformer is used to capture the temporal evolution of the sequence. Previous methods either use 2D depth images, sample skeletons points, or they require a huge amount of memory leading to the ability to process short sequences only. In this work, we show a competitive recognition rate and high memory efficiency by building our auto-encoder based on spiral convolutions, which are lightweight convolutions directly applied to mesh data with fixed topologies, and by modeling temporal evolution using attention, that can handle large sequences. The proposed method is evaluated on two 3D human action datasets: MoVi and BMLrub from the Archive of Motion Capture As Surface Shapes (AMASS). The results analysis shows the effectiveness of our method in 3D human action recognition while maintaining high memory efficiency.

## Repository Requirements

**PyTorch Version**: This code is designed for PyTorch version<1.1. To set up your environment, we recommend creating a virtual environment using Miniconda. You can install PyTorch within a conda environment using the following command (we tested it using PyTorch 1.10):

  ```
  $ conda install pytorch torchvision -c pytorch
  ```

  Next, install the remaining prerequisites by running:

  ```
  $ pip install -r requirements.txt
  ```

## Spiral Auto-Encoder

**NOTE: If you are using the Babel, Movi, or BMLrub datasets, you can disregard the Spiral Auto-Encoder steps outlined below. The extracted embeddings for these datasets are readily accessible [here](https://drive.google.com/drive/folders/1IandXYc7J0U0GnW8r48gnasPumt3P4x4?usp=sharing).**

To extract spatial embeddings from mesh data, we utilize the "Neural 3D Morphable Models: Spiral Convolutional Networks for 3D Shape Representation Learning and Generation." For a deeper understanding, please refer to the original work on GitHub at: [Neural3DMM](https://github.com/gbouritsas/Neural3DMM).

Our research relies on the use of the SMPL template and mesh decimation patterns during the training process. You can access the template and decimation patterns conveniently within the "NeuralSMPL/dataset/template" directory. If you wish to make adjustments to the template, mesh decimation, or filters sizes and number, you can find the relevant information here: [Neural3DMM](https://github.com/gbouritsas/Neural3DMM#data-organization).

### Data Organization:

Further details on data organization can be found here: [Data Organization](https://github.com/gbouritsas/Neural3DMM#data-organization).

### Training:

During the training phase, we exclusively use the training set, splitting it into 80% for training the Spiral Auto-Encoder and 20% for validation. We reserve the testing data for recognition purposes. You can execute the following commands to prepare the data:

```
$ python data_split.py
```

To use a PyTorch dataloader for training and testing, we split the data into separate files with:

```
$ python data_generation.py --root_dir=/path/to/data_root_dir --dataset=DFAUST --num_valid=100
```
Then, you can use Jupyter Notebook by running train.ipynb in the "NeuralSMPL" folder:

```
$ jupyter notebook train.ipynb
```

And set
```
$ args['mode'] = 'train'
```

### Testing:

To test the model, you can set 
```
$ args['mode'] = 'test'
```

## Data Embedding Extraction

to extract embedding from the whole dataset, set
```
$ args['mode'] = 'generate'
```


## Motion Transformer

For action recognition using the transformer, navigate to the "motion_transformer" folder and run the Jupyter Notebook named "transformer.ipynb".
