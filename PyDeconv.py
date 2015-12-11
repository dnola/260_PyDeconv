__author__ = 'davidnola'

import copy

import lasagne
import numpy as np
import theano
from skimage.util import view_as_blocks
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample


def normalize(image):
    print(image.max(),image.min())
    scale= abs(image.max() - image.min())
    image = (image - image.min())/scale

    return image

def get_switches(current,maxpool):
    # print("getting switches")
    current=np.array(current)
    switches = np.zeros((current.shape[0],current.shape[1],int(current.shape[2]/2)*2,int(current.shape[3]/2)*2))

    # print("before shape",switches.shape)
    for k1idx, k1 in enumerate(current):
        for k2idx, k2 in enumerate(k1):
            # print("before", k2[:4,:4])
            while k2.shape[0]%2!=0:
                k2=k2[:-1,:-1]
            blocks = view_as_blocks(k2,(2,2))
            for x in blocks:
                for y in x:
                    idx = np.unravel_index(y.argmax(), y.shape)
                    y.fill(0)
                    y[idx] = 1
            # print("after",k2[:4,:4])
            switches[k1idx,k2idx,:,:]=k2[:]
    # print("after shape",switches.shape)
    return switches

def forward_pass(current,weights,num_deconvs,maxouts):
    print("starting upward pass")
    # Forward pass
    switch_list = []
    for i in range(num_deconvs):
        print("up",i)
        W = weights[i]

        current = theano.tensor.nnet.conv.conv2d(current, W)
        current = lasagne.nonlinearities.rectify(current)

        if maxouts[i] != -1 and i!=num_deconvs-1:
            # print("Pooling...")
            maxpool_shape = tuple(list(map(lambda x:int(x), list(maxouts[i]))))
            maxpool = downsample.max_pool_2d(current, maxpool_shape, ignore_border=True)

            switch_list.append(get_switches(current.eval()[:],maxpool))
            # print("pre",current.eval().shape)
            current=maxpool
            # print(current.eval().shape)
        # current = current.eval()
        # print("current",current.shape)
        print("done",i)
    return current,switch_list

def downward_pass(current,weights,num_deconvs,maxouts,switch_list):
    print("starting downward pass")
    # Downward pass
    down = [np.swapaxes(W[:, :, ::-1, ::-1],0,1) for W in weights]

    for i in reversed(range(num_deconvs)):
        print("down",i)

        if maxouts[i] != -1 and i!=num_deconvs-1:
            # print("Pooling")
            try:
                current = upsample_switches(current.eval()[:],switch_list[i])
            except:
                current = upsample_switches(current[:],switch_list[i])

        current = lasagne.nonlinearities.rectify(current)
        current = theano.tensor.nnet.conv.conv2d(current, down[i],border_mode="full")
        print("done",i)
    return current


def deconv_image(image_in,weights_in,num_deconvs,kernel,maxouts):

    weights = copy.deepcopy(weights_in)
    image=copy.deepcopy(image_in)

    mod_im = np.swapaxes(image,0,2)
    mod_im = np.array(np.reshape(mod_im,(1,mod_im.shape[0],mod_im.shape[1],mod_im.shape[2])),dtype=np.float32)

    current = theano.shared(mod_im)

    if kernel != 'all':
        sz = weights[num_deconvs-1].shape
        for i in range(sz[0]):
            if i==kernel:
                continue
            weights[num_deconvs-1][i] = np.zeros((sz[1],sz[2],sz[3]),dtype=np.float32)


    current,switch_list = forward_pass(current,weights,num_deconvs,maxouts)

    current = downward_pass(current,weights,num_deconvs,maxouts,switch_list)

    current = current.eval()

    current = current.reshape((current.shape[1],current.shape[2],current.shape[3]))
    current = normalize(current)
    current = np.swapaxes(current,0,2)
    return current

def upsample_switches(current,switches):

    diff = int((switches.shape[-1] - current.shape[-1]*2)/4)
    current = np.pad(current, pad_width=((0,0),(0,0),(diff,diff),(diff,diff)), mode='constant', constant_values=0)
    # print("diff",diff)

    new = np.zeros((current.shape[0],current.shape[1],int(current.shape[2])*2,int(current.shape[3])*2))
    for k1idx, k1 in enumerate(current):
        for k2idx, k2 in enumerate(k1):
            # print(current[0,0,:4,:4])
            new[k1idx,k2idx,:,:] = k2.repeat(2, axis=0).repeat(2, axis=1)
            # print(current[0,0,:4,:4])

    diff = int((switches.shape[-1] - new.shape[-1])/2)
    new = np.pad(new, pad_width=((0,0),(0,0),(diff,diff),(diff,diff)), mode='constant', constant_values=0)
    return np.array(np.multiply(switches,new),dtype=np.float32)