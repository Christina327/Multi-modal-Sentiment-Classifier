import config

def update_config_with_args(args):
    config.TRAIN_WITH_LABEL_PATH = args.train_with_label_path
    config.TEST_WITHOUT_LABEL_PATH = args.test_without_label_path
    config.RAW_DATA_PATH = args.raw_data_path
    config.TRAIN_DATA_PATH = args.train_data_path
    config.TEST_DATA_PATH = args.test_data_path
    config.CACHE_MODEL_PATH = args.cache_model_path
    config.PREDICTION_PATH = args.prediction_path

    config.SEED = args.seed
    config.BATCH_SIZE = args.batch_size
    config.EPOCH = args.epoch
    assert 0 < args.lr < 1
    config.LR = args.lr

    config.TEXT_PRETRAINED_MODEL_NAME_OR_PATH = args.bert
    assert 0 < args.bert_lr < 1
    config.BERT_LR = args.bert_lr

    config.IMG_PRETRAINED_MODEL_NAME_OR_PATH = args.vit
    assert 0 < args.vit_lr < 1
    config.VIT_LR = args.vit_lrZ