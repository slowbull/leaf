import pickle
import numpy as np
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_path', help='saved path;', type=str, required=True)
    parser.add_argument('--num-rounds', help='number of rounds;', type=int, default=2000)
    parser.add_argument('--target', help='target point round;', type=int, default=2000)
    parser.add_argument('--eval-every', help='evaluate every ____ rounds;', type=int, default=100)

    args = parser.parse_args()

    parent_path = os.path.join('/dataset/hzy/FedLearning/result/checkpoints', args.save_path)

    file_path = os.path.join(parent_path, 'model_'+str(args.target))
    with open(file_path, 'rb') as f:
        target = pickle.load(f)

    print('Rounds  fedavg   fedsgd   fedavg-fedsgd  gradnorm')
    for k in range(0, args.num_rounds, args.eval_every):
        file_path = os.path.join(parent_path, 'fedavg_'+str(k))
        with open(file_path, 'rb') as f:
            fedavg_updates = pickle.load(f)
        file_path = os.path.join(parent_path, 'fedsgd_'+str(k))
        with open(file_path, 'rb') as f:
            fedsgd_updates = pickle.load(f)
        file_path = os.path.join(parent_path, 'model_'+str(k))
        with open(file_path, 'rb') as f:
            model = pickle.load(f)

        fedavg = [np.zeros(np.shape(v), dtype=float) for v in fedavg_updates[0][1]]
        
        for i, update in enumerate(fedavg_updates):
            for j, fedavg_val in enumerate(fedavg):
                fedavg[j] = np.add(fedavg_val, update[0] * update[1][j])

        fedsgd = [np.zeros(np.shape(v), dtype=float) for v in fedavg_updates[0][1]]
        for i, update in enumerate(fedsgd_updates):
            for j, fedsgd_val in enumerate(fedsgd):
                fedsgd[j] = np.add(fedsgd_val, update[0] * update[1][j])

        fedsgd_sigma = 0
        fedavg_sigma = 0
        diff_sigma = 0
        grad_l2 = 0
        for i in range(len(fedavg)):
            fedsgd_sigma += np.sum(np.multiply(fedsgd[i][1], -model[i]+target[i]))
            fedavg_sigma += np.sum(np.multiply(fedavg[i][1], -model[i]+target[i]))
            diff_sigma += np.sum(np.multiply(fedavg[i][1]-fedsgd[i][1], -model[i]+target[i]))
            grad_l2 += np.linalg.norm(fedsgd[i])

        print('{:03d}  {:.4f}  {:.4f}   {:.4f}  {:.4f}'.format(k, fedavg_sigma, fedsgd_sigma, diff_sigma, grad_l2))
       


if __name__ == '__main__':
    main()








