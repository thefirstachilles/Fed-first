import numpy as np
import gzip
import os
import platform
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'mnist':
            self.mnistDataSetConstruct(isIID)
        else:
            self.cifarDataSetConstruct(isIID)

    def cifarDataSetConstruct(self, isIID):
        train_images = None
        train_labels = None
        test_images = None
        test_labels = None
        data_dir = r'./data/cifar'
        for i in list(range(1,6)):
            train_cifar_path = os.path.join(data_dir, 'data_batch_{}'.format(i))
            with open(train_cifar_path, 'rb') as fo:
                cifar_content =  pickle.load(fo, encoding='bytes')
                if train_images is not None:
                    train_images = np.vstack((np.array(cifar_content[b'data']),train_images))
                else:
                    train_images = np.array( cifar_content[b'data'])
                if train_labels is not None:
                    train_labels = np.vstack((dense_to_one_hot(np.array(cifar_content[b'labels'])), train_labels))
                else:
                    train_labels = dense_to_one_hot(np.array(cifar_content[b'labels']))
        test_cifar_path = os.path.join(data_dir, 'test_batch')
        with open(test_cifar_path, 'rb') as fo:
            cifar_content = pickle.load(fo, encoding='bytes')
        test_images = np.array(cifar_content[b'data'])
        test_labels = dense_to_one_hot(np.array(cifar_content[b'labels']))

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        train_images = train_images.reshape(train_images.shape[0], 3, 32, 32)
        test_images = test_images.reshape(test_images.shape[0], 3, 32, 32)

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        # if isIID:
        #     order = np.arange(self.train_data_size)
        #     self.rng.shuffle(order)
        #     self.train_data = train_images[order]
        #     self.train_label = train_labels[order]
        # else:
        #     labels = np.argmax(train_labels, axis=1)
        #     order = np.argsort(labels)
        #     self.train_data = train_images[order]
        #     self.train_label = train_labels[order]
        self.train_data_data = test_images
        self.train_label = test_labels
        self.test_data = test_images
        self.test_label = test_labels

    def mnistDataSetConstruct(self, isIID):
        # data_dir = r'.\data\MNIST'
        data_dir = r'./data/MNIST/raw'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        # if isIID:
        #     order = np.arange(self.train_data_size)
        #     order = np.random.shuffle(order)
        #     self.train_data = train_images[order]
        #     self.train_label = train_labels[order]
        # else:
        #     labels = np.argmax(train_labels, axis=1)
        #     order = np.argsort(labels)
        #     self.train_data = train_images[order]
        #     self.train_label = train_labels[order]



        self.train_data = train_images[:10000]
        self.train_label = train_labels[:10000]
        self.test_data = test_images[:1000]
        self.test_label = test_labels[:1000]


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)

class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor

class Fed_server(object):
    def __init__(self):
        self.done = False
        self.accu_dict = []
        parent_dir = os.path.abspath(os.getcwd())
        self.dir_name = 'test_reward_accu.txt'
        path = os.path.join(parent_dir, 'res', self.dir_name)
        if not os.path.exists(path):
            os.mkdir(path)
        
        # 初始化网络
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = Mnist_2NN().to(self.dev)
        
      
        self.init_train_args()
        self.memo = []
        
        size = len(self.test_data_loader.dataset)
        num_batches = len(self.test_data_loader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, label in self.test_data_loader:
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = self.net(data)
                    test_loss += self.loss_func(preds, label).item()
                    correct += (preds.argmax(1) == label).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        self.last_loss = test_loss
        self.last_accu = correct

    def init_train_args(self):
        self.mnistDataSet = GetDataSet('mnist', True)
        self.loss_func = F.cross_entropy
        self.opti = optim.SGD(self.net.parameters(),lr=0.01)
        test_data = torch.tensor(self.mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(self.mnistDataSet.test_label), dim=1)
        train_data = torch.tensor(self.mnistDataSet.train_data)
        train_label = torch.argmax(torch.tensor(self.mnistDataSet.train_label), dim=1)
        self.train_data_loader = DataLoader(TensorDataset( train_data,  train_label), batch_size=10, shuffle=False)
        self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=10, shuffle=False)
        self.global_parameters = {}
        for key, var in self.net.state_dict().items():
            self.global_parameters[key] = var.clone()
        
        # 训练
    def train(self, round_index):
        size = len(self.train_data_loader.dataset)
        self.net.train()
        # self.net.load_state_dict(self.global_parameters, strict=True)
        for batch, (data, label) in enumerate(self.train_data_loader):
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                self.opti.step()
                self.opti.zero_grad()
                self.global_parameters = self.net.state_dict()
                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(data)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                
        # 测试
    def test(self, round_index):
        size = len(self.test_data_loader.dataset)
        num_batches = len(self.test_data_loader)
        self.net.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, label in self.test_data_loader:
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = self.net(data)
                    test_loss += self.loss_func(preds, label).item()
                    correct += (preds.argmax(1) == label).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        self.loss = test_loss
        self.global_accu = correct
        self.delta_accu = self.global_accu - self.last_accu
        self.delta_loss = self.last_loss - self.loss 
        # self.reward = self.delta_accu*((1-self.global_accu)**(-1))
        # self.reward = (self.loss-)*(round_index+1)/5
        self.reward = 1/self.loss
        
        
        self.memo.append({'test_loss':test_loss,'delta_loss':self.delta_loss,'global_accu':self.global_accu, 'delta_accu':self.delta_accu,'reward' : self.reward, 'last_loss':self.last_loss})
        self.last_accu =self.global_accu
        self.last_loss =self.loss
        with open(self.dir_name,  "w") as file:
                    file.write(json.dumps(self.memo))
                    file.close()
            
        
        
    
    
    
if __name__=="__main__":
    'test data set'
    server = Fed_server() # test NON-IID
    for round_index in range(20):
        server.train(round_index)
        server.test(round_index)
        
    
    

