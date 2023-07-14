import random, torch, os, sys
import numpy as np

sys.path.append('..')

ROOT_PATH = os.getcwd().replace('\\', '/') + '/'
TRAIN_WITH_LABEL_PATH = ROOT_PATH + 'data/train.txt'
TEST_WITHOUT_LABEL_PATH = ROOT_PATH + 'data/test_without_label.txt'
RAW_DATA_PATH = ROOT_PATH + 'data/raw/'
TRAIN_DATA_PATH = ROOT_PATH + 'data/input/train_data.json'
TEST_DATA_PATH = ROOT_PATH + 'data/input/test_data.json'
CACHE_MODEL_PATH = ROOT_PATH + 'cache/model'
PREDICTION_PATH = ROOT_PATH + 'cache/prediction.txt'
IMG_PRETRAINED_MODEL_NAME_OR_PATH = 'google/vit-base-patch16-224-in21k'
TEXT_PRETRAINED_MODEL_NAME_OR_PATH = 'bert-base-multilingual-cased'

SEED = 2023
BATCH_SIZE = 32
EPOCH = 10
LR = 1e-3
BERT_LR = 2e-5
VIT_LR = 2e-5

MAX_LEN = 128
FINE_TUNE = True


def setup_seed():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
