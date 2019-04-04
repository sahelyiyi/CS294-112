import os
import pickle
import subprocess
import numpy as np

from keras import models
from keras.layers import Dense, Dropout
from keras.models import load_model
from sklearn.model_selection import train_test_split

import methods
from config import get_cfg_defaults


def load_data(cfg, envname):
    with open(os.path.join(cfg.DATASETS.DATA_DIR, '%s.pkl' % envname), 'rb') as f:
        data = pickle.load(f)

    actions = data['actions']
    actions = actions.reshape(actions.shape[0], actions.shape[2])
    observations = data['observations']

    X_train, X_test, y_train, y_test = train_test_split(observations, actions, test_size=cfg.DATASETS.RATIO,
                                                        shuffle=cfg.DATASETS.SHUFFLE)
    return X_train, X_test, y_train, y_test


def build_model(cfg, input_size, output_size):
    # Build neural network
    model = models.Sequential()
    model.add(Dense(cfg.MODEL.NN.INPUT_LAYER[0]['units'], activation=cfg.MODEL.NN.INPUT_LAYER[0]['activation'], input_shape=(input_size,)))
    for hidden_layer in cfg.MODEL.NN.HIDDEN_LAYERS:
        model.add(Dense(hidden_layer['units'], activation=hidden_layer['activation']))
    model.add(Dense(output_size, activation=cfg.MODEL.NN.OUTPUT_LAYER[0]['activation']))
    model.add(Dropout(cfg.MODEL.NN.DROPOUT))

    # Compile model_data
    model.compile(optimizer=cfg.MODEL.NN.OPTIMIZER,
                  loss=cfg.MODEL.NN.LOSS,
                  metrics=cfg.MODEL.NN.METRICS)
    return model


def get_model_file_path(cfg, method_type, envname):
    return os.path.join(cfg.MODEL.NN.OUTPUT_DIR, '%s_%s' % (method_type, envname))


def train(cfg, model, model_file_path, X_train, y_train):
    model.fit(X_train, y_train,
              batch_size=cfg.INPUT.BATCH_SIZE,
              epochs=cfg.INPUT.EPOCHS,
              # callbacks=[plot_losses],
              verbose=1)
    model.save(model_file_path)


def test(model_file_path, X_test, y_test):
    model = load_model(model_file_path)
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--dagger', action='store_true')
    parser.add_argument('--render', action='store_true')

    args = parser.parse_args()

    # build the config
    cfg = get_cfg_defaults()
    cfg.freeze()

    X_train, X_test, y_train, y_test = load_data(cfg, args.envname)

    input_size = np.prod(X_train[0].shape)
    output_size = np.prod(y_train[0].shape)

    model = build_model(cfg, input_size, output_size)

    if args.dagger:
        model_file_path = get_model_file_path(cfg, cfg.MODEL.NN.DAGGER_FILE_NAME, args.envname)
        for i in range(cfg.MODEL.NN.DAGGER_EPOCHES):
            print('Dagger %d epoch' % i)
            train(cfg, model, model_file_path, X_train, y_train)
            subprocess.call(
                'python run_expert.py experts/%s.pkl %s --load_model %s' % (args.envname, args.envname, model_file_path))
            new_X_train, X_test, new_y_train, y_test = load_data(cfg, args.envname)
            X_train = np.concatenate((X_train, new_X_train))
            y_train = np.concatenate((y_train, new_y_train))

    else:
        model_file_path = get_model_file_path(cfg, cfg.MODEL.NN.BC_FILE_NAME, args.envname)
        train(cfg, model, model_file_path, X_train, y_train)
    test(model_file_path, X_test, y_test)


if __name__ == '__main__':
    main()
