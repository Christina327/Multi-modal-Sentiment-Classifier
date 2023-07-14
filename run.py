import os
import sys
import utils.config as config
from model.multi_classification import multi_run, multi_test, multi_pred
from model.img_classification import img_run, img_pred, img_test
from model.text_classification import text_predict, text_run, text_test
from utils.data_util import run as data_run
from utils.parser import parse_args
from utils.config_util import update_config_with_args


config.setup_seed()

if __name__ == '__main__':
    args = parse_args()
    update_config_with_args(args)

    if not (os.path.exists(config.TRAIN_DATA_PATH) and os.path.exists(config.TEST_DATA_PATH)):
        data_run()

    mode_actions = {
        'img_and_text': {'train': multi_run, 'test': multi_test, 'predict': multi_pred},
        'img_only': {'train': img_run, 'test': img_test, 'predict': img_pred},
        'text_only': {'train': text_run, 'test': text_test, 'predict': text_predict}
    }

    mode = args.mode
    if mode in mode_actions:
        actions = mode_actions[mode]
        for action, function in actions.items():
            if getattr(args, action):
                function()
