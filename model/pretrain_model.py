# coding = utf-8
# -*- coding:utf-8 -*-

from transformers import BertModel, BertTokenizer, ViTFeatureExtractor, ViTModel
from utils import config

config.setup_seed()


def getBert():
    return BertModel.from_pretrained(
        config.TEXT_PRETRAINED_MODEL_NAME_OR_PATH,
        config=config.TEXT_PRETRAINED_MODEL_NAME_OR_PATH
    )

def getTokenizer():
    return BertTokenizer.from_pretrained(config.TEXT_PRETRAINED_MODEL_NAME_OR_PATH)


def getViT():
    return ViTModel.from_pretrained(
        config.IMG_PRETRAINED_MODEL_NAME_OR_PATH,
        config=config.IMG_PRETRAINED_MODEL_NAME_OR_PATH
    )


def getExtractor():
    return ViTFeatureExtractor.from_pretrained(config.IMG_PRETRAINED_MODEL_NAME_OR_PATH)
