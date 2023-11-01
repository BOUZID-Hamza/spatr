from tqdm import tqdm
import numpy as np
import os, argparse



parser = argparse.ArgumentParser(description='Arguments for dataset split')
parser.add_argument('-r','--root_dir', type=str,
            help='Root data directory location, should be same as in neural3dmm.ipynb')
parser.add_argument('-d','--dataset', type=str, 
            help='Dataset name, Default is DFAUST')
parser.add_argument('-v','--num_valid', type=int, default=100, 
            help='Number of meshes in validation set, default 100')

args = parser.parse_args()


nVal = args.num_valid
root_dir = args.root_dir
dataset = args.dataset
name = ''

data = os.path.join(root_dir, dataset, 'preprocessed',name)


train = np.load(data+'/train.npy')
print("----done reading train npy file---- ")
if not os.path.exists(os.path.join(data,'points_train')):
    os.makedirs(os.path.join(data,'points_train'))

for i in tqdm(range(len(train))):
    np.save(os.path.join(data,'points_train','{0}.npy'.format(i)),train[i])
del train

files = []
for r, d, f in os.walk(os.path.join(data,'points_train')):
    for file in f:
        if '.npy' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(data,'paths_train.npy'),files)
print("----done saving train npy file---- ")

test = np.load(data+'/test.npy')
print("----done reading test npy file---- ")

if not os.path.exists(os.path.join(data,'points_val')):
    os.makedirs(os.path.join(data,'points_val'))
for i in tqdm(range(len(test))):
    np.save(os.path.join(data,'points_val','{0}.npy'.format(i)),test[i])
files = []
for r, d, f in os.walk(os.path.join(data,'points_val')):
    for file in f:
        if '.npy' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(data,'paths_val.npy'),files)
print("----done saving test npy file---- ")


