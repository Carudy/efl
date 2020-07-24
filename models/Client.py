import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models.Nets import CNNMnist, CNNCifar
from collections import defaultdict

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class FL_client(object):
    def __init__(self, args):
        if args.dataset == 'cifar':
            self.net = CNNCifar(args=args).to(args.device)
        else:
            self.net = CNNMnist(args=args).to(args.device)
        self.net.train()
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.lr)
        self.args      = args
        self.is_leader = False
        self.w_glob    = []
        self.x = self.gx = 0
        self.keys      = defaultdict(int)
        
    def set_data(self, dataset, idxs):
        self.data = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def load_state(self, state_dict):
        self.net.load_state_dict(state_dict)

    def train(self):
        epoch_loss = []
        for _ in range(self.args.local_ep):
            batch_loss = []
            for _, (images, labels) in enumerate(self.data):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # self.net.zero_grad()
                pred = self.net(images)
                loss = self.loss_func(pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return self.net.state_dict(), sum(epoch_loss) / len(epoch_loss)


# class LocalUpdate(object):
#     def __init__(self, args, dataset=None, idxs=None):
#         self.args = args
#         self.loss_func = nn.CrossEntropyLoss()
#         self.selected_clients = []
#         self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

#     def train(self, net):
#         net.train()
#         # train and update
#         optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9)

#         epoch_loss = []
#         for iter in range(self.args.local_ep):
#             batch_loss = []
#             for batch_idx, (images, labels) in enumerate(self.ldr_train):
#                 images, labels = images.to(self.args.device), labels.to(self.args.device)
#                 net.zero_grad()
#                 log_probs = net(images)
#                 loss = self.loss_func(log_probs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 if self.args.verbose and batch_idx % 10 == 0:
#                     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                         iter, batch_idx * len(images), len(self.ldr_train.dataset),
#                                100. * batch_idx / len(self.ldr_train), loss.item()))
#                 batch_loss.append(loss.item())
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#         return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
