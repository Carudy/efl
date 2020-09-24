import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch, copy, time
from sys import getsizeof

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Client import FL_client
from models.Nets import CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAdd
from models.test import test_img
from models.MPC import *

def show_acc(net, data_train, data_test, args):
    # net_test = copy.deepcopy(net)
    acc_train,  _ = test_img(net, data_train, args)
    acc_test,   _ = test_img(net, data_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    return acc_train.item()

def load_data(data_name, iid, num_users):
    if data_name == 'mnist':
        trans_mnist   = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        data_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        data_test  = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        if iid:
            dict_users = mnist_iid(data_train, num_users)
        else:
            dict_users = mnist_noniid(data_train, num_users)
    elif data_name == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        data_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        data_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if iid:
            dict_users = cifar_iid(data_train, num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    return data_train, data_test, dict_users

def DH(cid, client, lid, leader, shared_p, shared_g):
    client.x            =   np.random.randint(shared_p)
    client.gx           =   pow(shared_g, client.x, shared_p)
    leader.x            =   np.random.randint(shared_p)
    leader.gx           =   pow(shared_g, leader.x, shared_p)
    client.keys[lid]    =   pow(leader.gx, client.x, shared_p)
    leader.keys[cid]    =   pow(client.gx, leader.x, shared_p)

#********************************************************************************
if __name__ == '__main__':
    _DH     =   False
    _Crash  =   False
    _Drop   =   False

    args = args_parser()
    print('GPU: ', args.gpu, torch.cuda.is_available())
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    data_train, data_test, dict_users = load_data(args.dataset, args.iid, args.num_users)

    # build model
    if args.dataset == 'cifar':
        net_main = CNNCifar(args=args).to(args.device)
    else:
        net_main = CNNMnist(args=args).to(args.device)

    net_main.train()

    # copy weights
    w_glob = net_main.state_dict()
    w_zero = copy.deepcopy(w_glob)
    for i in w_zero: w_zero[i] *= 0.

    # clients
    # clients = [FL_client(args) for _ in range(n_clients)]
    clients = [FL_client(args) for _ in range(args.num_users)]
    for i, client in enumerate(clients): client.set_data(data_train, dict_users[i])
    
    # Select Leaders
    leaders = np.random.choice(range(args.num_users), 3, replace=False)
    # leaders = np.random.choice(range(n_clients), 3, replace=False)

    # DH set-up
    if _DH:
        shared_p, shared_g, cnt_comm = 10000079, 13, 0
        # dh_s = time.time()
        for lid, leader in [(i, clients[i]) for i in leaders]:
            for cid, client in enumerate(clients):
                if lid != cid:
                    DH(cid, client, lid, leader, shared_p, shared_g)
                    cnt_comm += 2           # send g^x and g^y
        # dh_e = time.time()
        # print('Set-up communication count: {},\tCost {:.3f} ms.'.format(cnt_comm, (dh_e - dh_s)*1000))

    # federated learning
    print("Start Learning:")
    m = max(int(args.frac * args.num_users), 1)

    plot_x      =   []
    plot_y      =   []
    # for crash_rate in [0.001, 0.002, 0.005, 0.01]:
    cnt_comm    =   0
    drop_rate = 0.1
    for epoch in range(args.epochs):
        loss_locals = []
        for i in leaders: clients[i].w_glob = []
        for i in np.random.choice(range(args.num_users), m, replace=False):
            # network delay / dropout
            if np.random.random() < drop_rate:
                continue
            clients[i].load_state(net_main.state_dict())
            w, loss = clients[i].train()
            w_divides = divide_dict(w)
            for i in range(len(leaders)):
                clients[leaders[i]].w_glob.append(copy.deepcopy(w_divides[i]))
                # while np.random.random() < crash_rate:
                #     cnt_comm += args.num_users*2
                # cnt_comm += 1

            loss_locals.append(loss)

        # secure aggregation
        w_glob = copy.deepcopy(w_zero)
        for leader in leaders:
            cnt_comm += 1
            _w = FedAvg(clients[leader].w_glob)
            w_glob = FedAdd(w_glob, _w)

        # copy weight to net_main
        net_main.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))

        if epoch in [4, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
            # show_acc(net_main, data_train, data_test, args)
            plot_x.append(epoch+1)
            plot_y.append(show_acc(net_main, data_train, data_test, args))
    # show_acc(net_main, data_train, data_test, args)
    plot_y = [round(i, 3) for i in plot_y]
    print(plot_x, plot_y)
