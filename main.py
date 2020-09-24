import numpy as np
from torchvision import datasets, transforms
import torch, copy, time, threading, random
import urllib.request
from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256

from utils.sampling import load_data
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

def gen_key(x, y, p):
    h = HMAC.new(b'demoFL', digestmod=SHA256)
    h.update(str(pow(x, y, p)).encode())
    return h.digest()

def DH(cid, client, lid, leader, shared_p, shared_g):
    client.x            =   random.randint(3, shared_p)
    client.gx           =   pow(shared_g, client.x, shared_p)
    leader.x            =   random.randint(3, shared_p)
    leader.gx           =   pow(shared_g, leader.x, shared_p)
    client.keys[lid]    =   gen_key(leader.gx, client.x, shared_p)
    leader.keys[cid]    =   gen_key(client.gx, leader.x, shared_p)

# for Bona-FedAvg
def DH_gen(client):
    client.x            =   random.randint(3, shared_p)     # s_u   for comm
    client.y            =   random.randint(3, shared_p)     # c_u   for Vec_uv
    client.gx           =   pow(shared_g, client.x, shared_p)
    client.gy           =   pow(shared_g, client.y, shared_p)
    client.bu           =   random.randint(3, shared_p)

def DH_send(i, j):
    clients[i].keys[j]    =   gen_key(clients[j].gx, clients[i].x, shared_p)
    clients[j].keys[i]    =   gen_key(clients[i].gx, clients[j].x, shared_p)

def server_send_recv():
    urllib.request.urlopen(urllib.request.Request(url))

def cal_poly(x, c):
    # global poly
    t = x * poly[0]
    for i in range(1, len(poly)):
        t = ((t * x) + poly[i]) % shared_p
    return (t * x + c) % shared_p

def t_out_of_n(client):
    for i in range(1, args.num_users): cal_poly(i, client.x)
    for i in range(1, args.num_users): cal_poly(i, client.bu)


#********************************************************************************
class Client_thread(threading.Thread):
    def __init__(self, net):
        super(Client_thread, self).__init__()
        self.net  =  net
        self.req  =  urllib.request.Request(url)

    def send(self):
        urllib.request.urlopen(self.req)

    def run(self):
        global clients, leaders, w_locals
        self.w, self.loss = self.net.train()

        if _Mode == 0:
            w_divides = divide_dict(self.w)
            for i in range(len(leaders)):
                clients[leaders[i]].w_glob.append(copy.deepcopy(w_divides[i]))

        elif _Mode == 1:
            w_locals.append(copy.deepcopy(self.w))
        

#********************************************************************************
if __name__ == '__main__':
    args = args_parser()
    global clients, leaders, w_locals, url, poly
    _Mode      =    0        # 0: 'demo'  1: 'bona'
    _DH        =    True
    _Crash     =    False
    _Drop      =    False
    n_leader   =    int(args.num_users * 0.1)
    w_locals   =    []
    shared_p, shared_g, cnt_comm = 6362166871434581, 13, 0    # For DH protocol
    url        =    'http://10.28.156.99:6789'        # simulate sending message
    poly       =    [random.randint(3, shared_p) for _ in range(args.num_users-1)]  # t-out-of-n poly

    print('DemoFL' if _Mode==0 else 'FedAvg')
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    ##### init global model
    if args.dataset == 'cifar':
        net_global = CNNCifar(args=args).to(args.device)
        args.num_channels = 3
        args.iid = True
    else:
        net_global = CNNMnist(args=args).to(args.device)
        args.num_channels = 1
        args.iid = False

    net_global.train()
    data_train, data_test, dict_users = load_data(args.dataset, args.iid, args.num_users)

    ##### copy w structure, create zero base
    w_glob = net_global.state_dict()
    w_zero = copy.deepcopy(w_glob)
    for i in w_zero: w_zero[i] *= 0.

    ##### Clients
    clients = [FL_client(args) for _ in range(args.num_users)]
    for i, client in enumerate(clients): client.set_data(data_train, dict_users[i])
    
#**********************************************************************************************
#**********************************************************************************************
    ##### Select Leaders
    leaders = np.random.choice(range(args.num_users), n_leader, replace=False)

    ##### Key exchange
    if _DH:
        ### Communication
        dh_s = time.time()
        if   _Mode==0:
            # send leaders to users
            # for _ in range(args.num_users): server_send_recv()
            for lid, leader in [(i, clients[i]) for i in leaders]:
                for cid, client in enumerate(clients):
                    if lid != cid:
                        DH(cid, client, lid, leader, shared_p, shared_g)
                        # server_send_recv()
        
        elif _Mode==1:
            # send shared_p etc. to users
            # for _ in range(args.num_users): server_send_recv()
            for client in clients: 
                DH_gen(client)
                # receive signing key
                # send  (...) to the server
                # server_send_recv()


        ### Computation
        if _Mode==1:
            for i in range(args.num_users):
                for j in range(i+1, args.num_users):
                    DH_send(i, j)
                t_out_of_n(clients[i])


        dh_e = time.time()
        print('Set-up Comm Avg Cost {:.3f} ms.'.format((dh_e - dh_s) * 1000 / args.num_users))

    exit()
    ##### federated learning
    print("Start Learning:")
    m = max(int(args.frac * args.num_users), 1)     # number of common-user

    plot_x      =   []
    plot_y      =   []
    # cnt_comm    =   0
    # drop_rate = 0.1

    ##### loop for epoch_max
    for epoch in range(args.epochs):
        w_locals    = []
        loss_locals = []
        ### clear users' weight
        for i in leaders: clients[i].w_glob = []

        ### choose m user to train local model
        train_users = np.random.choice(range(args.num_users), m, replace=False)
        workers = []
        for i in train_users:
            ### get global model from global
            clients[i].load_state(net_global.state_dict())
            workers.append(Client_thread(clients[i]))
            workers[-1].start()
            
        for td in workers: td.join()
        for td in workers: loss_locals.append(td.loss)

        # secure aggregation
        if  _Mode==0:
            w_glob = copy.deepcopy(w_zero)
            for leader in leaders:
                _w = FedAvg(clients[leader].w_glob)
                w_glob = FedAdd(w_glob, _w)

        elif _Mode==1:
            w_glob = FedAvg(w_locals)

        # copy weight to net_global
        net_global.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))


        ### Plot data
        if epoch in [4, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
            # show_acc(net_global, data_train, data_test, args)
            plot_x.append(epoch+1)
            plot_y.append(show_acc(net_global, data_train, data_test, args))

    # show_acc(net_global, data_train, data_test, args)
    print(plot_y)