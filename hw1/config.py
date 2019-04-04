import os
from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# Model Configs
# ---------------------------------------------------------------------------- #

_C.MODEL = CN()

# ---------------------------------------------------------------------------- #
# __NN Configs
# ---------------------------------------------------------------------------- #

_C.MODEL.NN = CN()

_C.MODEL.NN.INPUT_LAYER = [{'units': 128, 'activation': 'relu'},]

_C.MODEL.NN.HIDDEN_LAYERS = [
    {'units': 64, 'activation': 'relu'},
    {'units': 32, 'activation': 'relu'},
]

_C.MODEL.NN.OUTPUT_LAYER = [{'activation': 'linear'},]

_C.MODEL.NN.DROPOUT = 0.2

_C.MODEL.NN.OPTIMIZER = 'adam'
_C.MODEL.NN.LOSS = 'mse'
_C.MODEL.NN.METRICS = ['accuracy']

_C.MODEL.NN.OUTPUT_DIR = 'model_data'

_C.MODEL.NN.BC_FILE_NAME = 'behavioral_cloning'
_C.MODEL.NN.DAGGER_FILE_NAME = 'dagger'
_C.MODEL.NN.DAGGER_EPOCHES = 10

# ---------------------------------------------------------------------------- #
# Input Pipeline Configs
# ---------------------------------------------------------------------------- #

_C.INPUT = CN()
_C.INPUT.BATCH_SIZE = 100
_C.INPUT.EPOCHS = 50

# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.DATA_DIR = 'expert_data'                  # path to dataset
_C.DATASETS.SHUFFLE = True                              # load in shuffle fashion
_C.DATASETS.RATIO = 0.2

# ---------------------------------------------------------------------------- #


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()