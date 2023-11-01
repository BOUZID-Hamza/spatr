import numpy as np
import trimesh
import os
from glob import glob
from tqdm import tqdm
import random

data_path= r'D:\Babel\train'

train_BML= []
test_BML = []
print('---loading paths---')
paths = glob(os.path.join(data_path,'*','*.ply'))
print(len(paths))

train_paths = paths[:int(len(paths)*0.8)]
test_paths = paths[int(len(paths)*0.8):]

for i,path in enumerate(tqdm(train_paths)):
    data_loaded = trimesh.load(path, process=False)
    vertices = data_loaded.vertices
    train_BML.append(vertices)
print(np.shape(train_BML))
if not os.path.exists(os.path.join('babel', 'preprocessed')):
    os.makedirs(os.path.join('babel', 'preprocessed'))
np.save(os.path.join('babel', 'preprocessed','train.npy'), train_BML)
del train_BML

for i,path in enumerate(tqdm(test_paths)):
    data_loaded = trimesh.load(path, process=False)
    vertices = data_loaded.vertices
    test_BML.append(vertices)
#print(np.shape(test_BML))
np.save(os.path.join('babel', 'preprocessed','test.npy'), test_BML)
print("done")


