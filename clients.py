import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt


class client(object):
    def __init__(self, trainDataSet, testDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.test_ds = testDataSet
        self.test_dl = None

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()
        return Net.state_dict()
    # def testLocal(self):

    #     return

    def local_val(self, Net, global_parameters,lossFun):
        with torch.no_grad():
            Net.load_state_dict(global_parameters, strict=True)
            # self.test_dl = DataLoader(self.test_ds)
            self.test_dl = DataLoader(self.train_ds)
            num_batches = len(self.test_dl)
            size = len(self.test_dl.dataset)
            test_loss, correct = 0, 0
            for data, label in self.test_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                test_loss += lossFun(preds, label).item()
                correct += (preds.argmax(1) == label).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        # Net.load_state_dict(global_parameters, strict=True)
        # self.test_dl = DataLoader(self.test_ds)
        # with torch.no_grad():
        #     sum_accu = 0
        #     num = 0
        #     for data, label in self.test_dl:
        #         data, label = data.to(self.dev), label.to(self.dev)
        #         preds = Net(data)
        #         preds = torch.argmax(preds, dim=1)
        #         sum_accu += (preds == label).float().mean()
        #         num += 1
        #     accu = sum_accu / num
        return test_loss


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev, rng):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}

        self.test_data_loader = None
        self.rng = rng

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid, self.rng)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
        shards_id = self.rng.permutation(mnistDataSet.train_data_size // shard_size)
        test_size = mnistDataSet.test_data_size // self.num_of_clients 
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            
            # 随机抽取1000张 减小运算量
            permutation_order = self.rng.permutation(local_data.shape[0])[:100]
            local_data, local_label = local_data[permutation_order], local_label[permutation_order]
            local_label = np.argmax(local_label, axis=1)
            local_test_data = mnistDataSet.test_data[i*test_size: i*test_size + test_size][:100]
            local_test_label = mnistDataSet.test_label[i*test_size: i*test_size + test_size][:100]
            local_test_label = np.argmax(local_test_label, axis=1)
            # 处理图像
            # 按照一定比例处理，比方i=0 不处理 其他处理百分之i*10
            if i > 0:
                process_order = np.arange(0,i*10,dtype=int)
                for x in process_order:
                    # process_lable = local_label[x]
                    process_data = local_data[x].reshape(28,28,1)
                    process_data = gaussian_filter(process_data, sigma=(2,2,0))
                    local_data[x] = process_data.reshape(-1,)
                    
                    # process_lable = local_test_label[x]
                    process_data = local_test_data[x].reshape(28,28,1)
                    process_data = gaussian_filter(process_data, sigma=(2,2,0))
                    local_test_data[x] = process_data.reshape(-1,)
            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), TensorDataset(torch.tensor(local_test_data), torch.tensor(local_test_label)), self.dev)
            self.clients_set['client{}'.format(i)] = someone

if __name__=="__main__":
    rng2 = np.random.default_rng(seed=100)
    MyClients = ClientsGroup('cifar', True, 100, 1, rng2)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])


