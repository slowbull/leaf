"""Script to run the baselines."""

import argparse
import importlib
import numpy as np
import os
import sys
import tensorflow as tf

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS, SIM_TIMES
from client import Client
from server import Server
from model import ServerModel
from model_mom import ServerModel as ServerModel_mom

from utils.constants import DATASETS
from utils.model_utils import read_data

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'


def main():

    args = parse_args()

    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)

    save_path = os.path.join('/dataset/hzy/FedLearning/result/checkpoints', args.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # Create 2 models
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)
    tf.reset_default_graph()
    client_model = ClientModel(*model_params)
    if args.mom == 0:
        server_model = ServerModel(ClientModel(*model_params))
    else:
        server_model = ServerModel_mom(ClientModel(*model_params))


    # Create server
    server = Server(server_model)

    # Create clients
    clients = setup_clients(args.dataset, client_model)
    print('%d Clients in Total' % len(clients))

    print('Rounds Train_acc_avg  p_10  p_90  Train_loss_avg  p_10  p_90  Test_acc_avg  p_10 p_90  Test_loss_avg  p_10 p_90')
    # Test untrained model on all clients
    stat_metrics = server.test_model(clients)
    all_ids, all_groups, all_num_samples = server.get_clients_test_info(clients)
    metrics_writer.print_metrics(0, all_ids, stat_metrics, all_groups, all_num_samples, STAT_METRICS_PATH)
    print_metrics(stat_metrics, all_num_samples, 0)


    # Simulate training
    for i in range(num_rounds):
        #print('--- Round %d of %d: Training %d Clients ---' % (i+1, num_rounds, clients_per_round))

        # Select clients to train this round
        server.select_clients(online(clients), num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples = server.get_clients_test_info()


        # simulate mini_batch grad
        _ = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=1.0)
        if (i) % eval_every == 0 or (i + 1) == num_rounds:
            filename = os.path.join(save_path, 'fedsgd_'+str(i))
            server.save_update(filename)
            filename = os.path.join(save_path, 'model_'+str(i))
            server.save_model(filename)
            

        # Simulate server model training on selected clients' data
        sys_metics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)
        metrics_writer.print_metrics(i, c_ids, sys_metics, c_groups, c_num_samples, SYS_METRICS_PATH)

        if (i) % eval_every == 0 or (i + 1) == num_rounds:
            filename = os.path.join(save_path, 'fedavg_'+str(i))
            server.save_update(filename)

        # Update server model
        server.update_model()

        # Test model on all clients
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            stat_metrics = server.test_model(clients)
            metrics_writer.print_metrics(i, all_ids, stat_metrics, all_groups, all_num_samples, STAT_METRICS_PATH)
            print_metrics(stat_metrics, all_num_samples, i+1)

    # Save server model
    save_model(server_model, save_path, args.model)

    # Close models
    server_model.close()
    client_model.close()


def online(clients):
    """We assume all users are always online."""
    return clients


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    required=True)
    parser.add_argument('-save_path',
                    help='path of checkpoints;',
                    type=str,
                    required=True)
    parser.add_argument('-model',
                    help='name of model;',
                    type=str,
                    required=True)
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=-1)
    parser.add_argument('--eval-every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=-1)
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=int,
                    default=-1)
    parser.add_argument('--batch_size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--minibatch',
                    help='None for FedAvg, else fraction;',
                    type=float,
                    default=None)
    parser.add_argument('--num_epochs',
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=1)
    parser.add_argument('-t',
                    help='simulation time: small, medium, or large;',
                    type=str,
                    choices=SIM_TIMES,
                    default='large')
    parser.add_argument('-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=-1,
                    required=False)
    parser.add_argument('-mom',
                    help='momentum term;',
                    type=float,
                    default=0.,)
    return parser.parse_args()


def setup_clients(dataset, model=None):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    train_data_dir = os.path.join('/dataset/hzy/FedLearning', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('/dataset/hzy/FedLearning', 'data', dataset, 'data', 'test')

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    if len(groups) == 0:
        groups = [[] for _ in users]
    all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return all_clients


def save_model(server_model, save_path, model):
    """Saves the given server model on checkpoints/dataset/model.ckpt."""
    # Save server model
    #ckpt_path = os.path.join('checkpoints', save_path)
    ckpt_path = save_path
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server_model.save(os.path.join(ckpt_path, '%s.ckpt' % model))
    print('Model saved in path: %s' % save_path)


def print_metrics(metrics, weights, rounds):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    print('{} '.format(rounds), end='')
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        #print('%s: %g, 10th percentile: %g, 90th percentile %g' \
        #      % (metric,
        #         np.average(ordered_metric, weights=ordered_weights),
        #         np.percentile(ordered_metric, 10),
        #         np.percentile(ordered_metric, 90)))
        print('{:.4f}   {:.4f}   {:.4f} '.format(\
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 90)), end='')

    print(' ')

if __name__ == '__main__':
    main()
