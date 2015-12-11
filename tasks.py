
import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
from invoke import *
from PyDeconv import deconv_image


@task(name="display")
def display_images(image_path="../train/1456.png",net_path='net_maxout.h5'):
    image =  scipy.ndimage.imread(image_path)
    weights_h5 = h5py.File(net_path, 'r')

    weights = []
    maxouts = []
    for i in range(99):
        try:
            weights.append(np.array(weights_h5[str(i)],dtype=np.float32))
            try:
                maxouts.append((weights_h5[str(i)].attrs['maxpool_x'],weights_h5[str(i)].attrs['maxpool_y']))
            except:
                maxouts.append(-1)
        except:
            print("Ending on:",i)
            break


    width = np.min([weights[j].shape[0] for j in range(len(weights))])
    plt.figure(figsize = (len(weights),width))
    gs1 = gridspec.GridSpec( len(weights),width)
    gs1.update(wspace=0.001, hspace=0.001)

    images = []
    idx = 0

    for j in range(len(weights)):
        for i in range(width):
            im = deconv_image(image,weights,j+1,i,maxouts)
            ax1 = plt.subplot(gs1[idx])
            plt.axis('on')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            plt.imshow(im)
            idx+=1

    plt.show()


@task(name="generate")
def generate_images(image_path="../train/21.png",net_path='net_maxout.h5'):
    image =  scipy.ndimage.imread(image_path)
    weights_h5 = h5py.File(net_path, 'r')

    weights = []
    maxouts = []
    for i in range(99):
        try:
            weights.append(np.array(weights_h5[str(i)],dtype=np.float32))
            try:
                maxouts.append((weights_h5[str(i)].attrs['maxpool_x'],weights_h5[str(i)].attrs['maxpool_y']))
            except:
                maxouts.append(-1)
        except:
            print("Ending on:",i)
            break

    output = h5py.File('output.h5', 'w')

    for j in range(len(weights)):
        layer = output.create_group(str(j))
        for i in range(weights[j].shape[1]):
            im = deconv_image(image,weights,j+1,i,maxouts)
            output[str(j)].create_dataset(str(i), data=im)


