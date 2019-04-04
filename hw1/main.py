import os
import pickle
import subprocess
import numpy as np

from sklearn.model_selection import train_test_split

from config import get_cfg_defaults

from model import build_model, train, test


def load_data(cfg, envname):
    with open(os.path.join(cfg.DATASETS.DATA_DIR, '%s.pkl' % envname), 'rb') as f:
        data = pickle.load(f)
    return data


def split_data(cfg, data):
    actions = data['actions']
    actions = actions.reshape(actions.shape[0], actions.shape[2])
    observations = data['observations']

    X_train, X_test, y_train, y_test = train_test_split(observations, actions,
                                                        test_size=cfg.DATASETS.RATIO, shuffle=cfg.DATASETS.SHUFFLE)

    return X_train, X_test, y_train, y_test


def get_model_file_path(cfg, method_type, envname):
    return os.path.join(cfg.MODEL.NN.OUTPUT_DIR, '%s_%s' % (method_type, envname))


def train_dagger(cfg, model, model_file_path, envname, X_train, y_train, num_rollouts):
    rewards_mean = []
    rewards_std = []
    for i in range(cfg.MODEL.NN.DAGGER_EPOCHES):
        print('Dagger epoch number %d' % (i + 1))
        train(cfg, model, model_file_path, X_train, y_train)

        subprocess.call(
            'python run_expert.py experts/%s.pkl %s --num_rollouts %d --load_model %s' %
            (envname, envname, num_rollouts, model_file_path),
            shell=True)

        data = load_data(cfg, envname)

        returns = data['returns']
        rewards_mean.append(np.mean(returns))
        rewards_std.append(np.std(returns))

        new_X_train, X_test, new_y_train, y_test = split_data(cfg, data)
        X_train = np.concatenate((X_train, new_X_train))
        y_train = np.concatenate((y_train, new_y_train))

    return rewards_mean, rewards_std


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--dagger', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=2,
                        help='Number of expert roll outs in dagger mode')

    args = parser.parse_args()

    # build the config
    cfg = get_cfg_defaults()
    cfg.freeze()

    data = load_data(cfg, args.envname)
    X_train, X_test, y_train, y_test = split_data(cfg, data)

    input_size = np.prod(X_train[0].shape)
    output_size = np.prod(y_train[0].shape)

    model = build_model(cfg, input_size, output_size)

    if args.dagger:
        model_file_path = get_model_file_path(cfg, cfg.MODEL.NN.DAGGER_FILE_NAME, args.envname)
        rewards_mean, rewards_std = train_dagger(cfg, model, model_file_path, args.envname, X_train, y_train, args.num_rollouts)
        print('mean of rewards per iterations is', rewards_mean)
        print('std of rewards per iterations is', rewards_std)

    else:
        model_file_path = get_model_file_path(cfg, cfg.MODEL.NN.BC_FILE_NAME, args.envname)
        train(cfg, model, model_file_path, X_train, y_train)

    test(model_file_path, X_test, y_test)


if __name__ == '__main__':
    main()
