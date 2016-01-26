__author__ = 'davidnola'

import glob
import pickle

import lasagne
import numpy as np
import pandas
import scipy.ndimage
import theano
from lasagne import layers
from lasagne.init import GlorotUniform
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

import h5py

theano.gof.cc.get_module_cache().clear()
######################### Start Helper Class/Function Definitions ##########################


def float32(k):
    return np.cast['float32'](k)


class EarlyStopping(object):
    def __init__(self, patience=20):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

class AdjustVariable(object):
    def __init__(self, name):
        self.name = name


    def __call__(self, nn, train_history):
        getattr(nn, self.name).set_value(getattr(nn, self.name)/2.0)


def dump_weights(net,filename):
    input_shape=net.layers_[0].shape
    print("input shape:", input_shape)

    weights_hdf5 = h5py.File(filename, 'w')

    num_layers = 0
    for idx, l in enumerate(net.layers_[1:]):
        try:
            W = np.array(net.layers_[l].W.eval(),dtype=np.float32)
            print(W.shape)
            if(len(W.shape)<4):
                print("Done")
                break
            weights_hdf5.create_dataset(str(num_layers),data=W)
            num_layers+=1
        except Exception as e:
            print(e)
            continue

    # weights_hdf5["0"].attrs['activation'] = "relu"
    weights_hdf5["0"].attrs['maxpool_x'] = 2
    weights_hdf5["0"].attrs['maxpool_y'] = 2
    # weights_hdf5["1"].attrs['maxpool_x'] = 2
    # weights_hdf5["1"].attrs['maxpool_y'] = 2

    weights_hdf5.close()


######################### Start Preprocessing ##########################

train_glob = 'C:\\Users\\davidnola\\Documents\\Programming\\PyDeconv\\train\\*.png'
train_labels = 'trainLabels.csv'
test_glob = 'C:\\Users\\davidnola\\Documents\\Programming\\PyDeconv\\test\\*.png'



label_file = pandas.read_csv(train_labels)
labels = label_file['label'].values

print("loading train...")
X = []
y = []
for i,f in enumerate(glob.glob(train_glob)):
    g_id = int((f.split('\\')[-1]).split('.')[0])
    im = scipy.ndimage.imread(f)
    im = np.swapaxes(im,0,2)
    X.append(im)
    y.append(labels[g_id-1])
X=np.array(X,dtype=np.float32)

print("loading test...")
X_test = []
X_test_ids = []
for i,f in enumerate(glob.glob(test_glob)[:10]):
    im = scipy.ndimage.imread(f)
    im = np.swapaxes(im,0,2)
    X_test.append(im)
    g_id = int(((f.split('\\')[-1]).split('.')[0]))
    X_test_ids.append(g_id)

X_test=np.array(X_test,dtype=np.float32)

print(X.shape,len(X_test_ids))

label_encoder = LabelEncoder()
one_hot = OneHotEncoder()
label_encoder.fit(y)
one_hot.fit(list(map(lambda x:[x],label_encoder.transform(y))))
stack_encoder = lambda x: one_hot.transform(list(map(lambda x:[x],label_encoder.transform(x)))).toarray()
y=np.array(stack_encoder(y),dtype=np.float32)

######################### Start Training ##########################

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('maxout1', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('conv4', layers.Conv2DLayer),
        ('maxout2', layers.MaxPool2DLayer),
        # ('conv5', layers.Conv2DLayer),
        ('dense', layers.DenseLayer),
        ('dense2', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None,3, 32,32),

    conv1_num_filters=16, conv1_filter_size=(3, 3), conv1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2_num_filters=16, conv2_filter_size=(3, 3), conv2_nonlinearity=lasagne.nonlinearities.rectify,
    conv3_num_filters=16, conv3_filter_size=(3, 3), conv3_nonlinearity=lasagne.nonlinearities.rectify,

    maxout1_pool_size=2,
    maxout2_pool_size=2,

    dense_num_units=256,dense_W=GlorotUniform(),
    dense2_num_units=256,dense2_W=GlorotUniform(),

    output_nonlinearity=lasagne.nonlinearities.softmax, output_num_units=len(y[0]),

    on_epoch_finished=[EarlyStopping(),AdjustVariable('update_learning_rate')],

    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.01)),
    update_momentum=theano.shared(float32(0.90)),

    regression=True,
    max_epochs=750,
    verbose=10,
    )


print("Fitting net...")
net.fit(X,y)
print(len(net.layers_))

######################### Save Network ##########################

dump_weights(net,'net.h5')


######################### Generate Kaggle Submission ##########################


preds = label_encoder.inverse_transform(list(map(np.argmax,net.predict(X_test))))

final_str = "id,label"
for idx,i in enumerate(X_test_ids):
    final_str+="\n"
    final_str+=str(i)
    final_str+=","
    final_str+=preds[idx]

with open('network_output.csv','w') as f:
    f.write(final_str)
