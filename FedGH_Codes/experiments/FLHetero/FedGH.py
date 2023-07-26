import argparse
import copy
import csv
import json
import logging
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
import sys



import numpy as np
import torch
import torch.utils.data
from tqdm import trange

from experiments.FLHetero.Models.CNNs import CNN_1
from experiments.FLHetero.Models.FC import FC

from experiments.FLHetero.node import BaseNodes
from experiments.utils import get_device, set_logger, set_seed, str2bool


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 





random.seed(2022)

def test_acc(net, testloader,criteria):
    net.eval()
    with torch.no_grad():
        test_acc = 0
        num_batch = 0

        for batch in testloader:
            num_batch += 1
            # batch = next(iter(testloader))
            img, label = tuple(t.to(device) for t in batch)
            pred, _ = net(img)
            test_loss = criteria(pred, label)
            test_acc += pred.argmax(1).eq(label).sum().item() / len(label)
        mean_test_loss = test_loss / num_batch
        mean_test_acc = test_acc / num_batch
    return mean_test_loss, mean_test_acc



def train(data_name: str, data_path: str, classes_per_node: int, num_nodes: int, fraction: float,
          steps: int, epochs: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, hyper_hid: int,
          n_hidden: int, n_kernels: int, bs: int, device, eval_every: int, save_path: Path,
          seed: int, total_classes: int) -> None:

    ###############################
    # init nodes, hnet, local net #
    ###############################
    nodes = BaseNodes(data_name, data_path, num_nodes, classes_per_node=classes_per_node,
                      batch_size=bs)

    # -------compute aggregation weights-------------#
    train_sample_count = nodes.train_sample_count
    eval_sample_count = nodes.eval_sample_count
    test_sample_count = nodes.test_sample_count

    client_sample_count = [train_sample_count[i] + eval_sample_count[i] + test_sample_count[i] for i in
                           range(len(train_sample_count))]
    # -----------------------------------------------#


    print(data_name)
    if data_name == "cifar10":
        net = CNN_1(n_kernels=n_kernels)
        net_FC = FC(in_dim=500, out_dim=10)
    elif data_name == "cifar100":
        net = CNN_1(n_kernels=n_kernels, out_dim=100)
        net_FC = FC(in_dim=500, out_dim=100)
    elif data_name == "mnist":
        net = CNN_1(n_kernels=n_kernels)
    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100']")


    net = net.to(device)
    net_FC = net_FC.to(device)

    init_Gnet_paras = copy.deepcopy(net.state_dict()) 
    #
    # lhn_i_flatten = [i.reshape(-1) for i in init_Gnet_paras.values()]
    # lhn_i = torch.cat(lhn_i_flatten, dim=0)

    ##################
    # init optimizer #
    ##################

    optimizers = {
        'sgd': torch.optim.SGD(params=net.parameters(), lr=lr, momentum=0.9, weight_decay=wd),
        'adam': torch.optim.Adam(params=net.parameters(), lr=lr)
    }
    optimizer = optimizers[optim]
    criteria = torch.nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
    #                                            milestones=[int(steps * 0.56), int(steps * 0.78)],
    #                                            gamma=0.1, last_epoch=-1)

    ################
    # init metrics #
    ################
    step_iter = trange(steps)

    Gnet_paras = init_Gnet_paras
    PM_acc = defaultdict()
    PMs = defaultdict()
    Data_Distirbutions = defaultdict()
    Protos = defaultdict()
    Global_Proto = defaultdict()

    for i in range(num_nodes):
        PM_acc[i] = 0
        PMs[i] = init_Gnet_paras
        Protos[i] = defaultdict(list)
        Data_Distirbutions[i] = defaultdict(int)





    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(str(save_path / f"FedGH_N{num_nodes}_C{fraction}_R{steps}_E{epochs}_B{bs}_NonIID_{data_name}_class_{classes_per_node}.csv"), 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')


        for step in step_iter:  # step is round 
            round_id = step
            frac = fraction 
            select_nodes = random.sample(range(num_nodes), int(frac * num_nodes))

            all_local_trained_loss = []
            all_local_trained_acc = []
            all_global_loss = []
            all_global_acc = []
            results = []

            Protos_Mean = defaultdict()
            for i in select_nodes:
                Protos_Mean[i] = defaultdict()

            logging.info(f'#----Round:{step}----#')
            for c in select_nodes:
                node_id = c
                print(f'client id: {node_id}')

                # net.load_state_dict(PMs[node_id])

                if round_id == 0:
                    net.load_state_dict(init_Gnet_paras)
                else:
                    # directly use global shared predictor
                    net_paras = dict(PMs[node_id], **Global_header)
                    net.load_state_dict(net_paras)



                # evlaute GM
                # global_loss,  global_acc = test_acc(net, nodes.test_loaders[node_id], criteria)
                # all_global_loss.append(global_loss.cpu().item())
                # all_global_acc.append(global_acc)
                all_global_loss.append(0)
                all_global_acc.append(0)

                # local training
                # net = localtraining(epochs, net, nodes.train_loaders[node_id], optimizer, criteria, node_id)
                net.train()
                for i in range(epochs):
                    for j, batch in enumerate(nodes.train_loaders[node_id], 0):
                        # print(f'batch is {j}')
                        img, label = tuple(t.to(device) for t in batch)

                        optimizer.zero_grad()

                        pred, rep = net(img)

                        loss = criteria(pred, label)

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 50)
                        optimizer.step()

                # PMs[node_id] = copy.deepcopy(net.state_dict())
                full_net = copy.deepcopy(net.state_dict())
                del_keys = list(full_net.keys())[-2:]
                for key in del_keys:
                    full_net.pop(key)


                PMs[node_id] = copy.deepcopy(full_net)

                # evaluate trained local model
                trained_loss, trained_acc = test_acc(net, nodes.test_loaders[node_id], criteria)
                all_local_trained_loss.append(trained_loss.cpu().item())
                all_local_trained_acc.append(trained_acc)
                PM_acc[node_id] = trained_acc

                logging.info(f'Round {step} | client {node_id} acc: {PM_acc[node_id]}')

              
                # scheduler.step()
                # print('\t last_lr:', scheduler.get_last_lr())

                # collect local NN parameters
                proto_mean = defaultdict(list)

                for j, batch in enumerate(nodes.train_loaders[node_id], 0):
                    img, label = tuple(t.to(device) for t in batch)

                    pred, rep = net(img)

                    owned_classes = label.unique().detach().cpu().numpy()
                    for cls in owned_classes:
                        filted_reps = list(map(lambda x: x[0], filter(lambda x: x[1] == cls, zip(rep, label))))
                        sum_filted_reps = filted_reps[0].detach()
                        for f in range(1, len(filted_reps)):
                            sum_filted_reps = sum_filted_reps + filted_reps[f].detach()

                        mean_filted_reps = sum_filted_reps / len(filted_reps)
                        proto_mean[cls].append(mean_filted_reps)

                for cls, protos in proto_mean.items():
                    sum_proto = protos[0]
                    for m in range(1, len(protos)):
                        sum_proto = sum_proto + protos[m]
                    mean_proto = sum_proto/len(protos)

                    Protos_Mean[node_id][cls] = mean_proto

            mean_trained_loss = round(np.mean(all_local_trained_loss), 4)
            mean_trained_acc = round(np.mean(all_local_trained_acc), 4)
            mean_global_loss = round(np.mean(all_global_loss), 4)
            mean_global_acc = round(np.mean(all_global_acc), 4)
            results.append([mean_global_loss, mean_global_acc, mean_trained_loss, mean_trained_acc] + [round(i,4) for i in PM_acc.values()])
            mywriter.writerows(results)
            file.flush()

            logging.info(f'Round:{step} | mean_global_loss:{mean_global_loss} | mean_global_acc:{mean_global_acc} | mean_trained_loss:{mean_trained_loss} | mean_trained_acc:{mean_trained_acc}')


            # aggregate
            net_FC.train()
            for c in select_nodes:
                for cls, rep in Protos_Mean[c].items():
                    pred_server = net_FC(rep)
                    loss = criteria(pred_server.view(1,-1), torch.tensor(cls).view(1).to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net_FC.parameters(), 50)
                    optimizer.step()

            Global_header = copy.deepcopy(net_FC.state_dict())


            logging.info(f'Global Proto is updated after aggregation')

        logging.info('Federated Learning has been successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Federated Learning with Lookahead experiment"
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="cifar100", choices=['cifar10', 'cifar100', 'mnist'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--total-classes", type=str, default=100)
    parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")
    parser.add_argument("--num-nodes", type=int, default=100, help="number of simulated nodes")
    parser.add_argument("--fraction", type=int, default=0.1, help="number of sampled nodes in each round")

    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10, help="number of inner steps")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-3, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="Results/temp", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu) 

    if args.data_name == 'cifar10':
        args.classes_per_node = 2 # 2, 4, 6, 8, 10
    elif args.data_name == 'cifar100':
        args.classes_per_node = 10 # 30, 50, 70, 90, 100
    else:
        args.classes_per_node = 2

    train(
        data_name=args.data_name,
        data_path=args.data_path,
        classes_per_node=args.classes_per_node,
        total_classes = args.total_classes,
        num_nodes=args.num_nodes,
        fraction=args.fraction,
        steps=args.num_steps,
        epochs=args.epochs,
        optim=args.optim,
        lr=args.lr,
        inner_lr=args.inner_lr,
        embed_lr=args.embed_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        embed_dim=args.embed_dim,
        hyper_hid=args.hyper_hid,
        n_hidden=args.n_hidden,
        n_kernels=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path,
        seed=args.seed
    )
