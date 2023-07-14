# coding = utf-8
# -*- coding:utf-8 -*-
import json, torch, sys

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import model.pretrain_model as pretrain_model
import config

config.setup_seed()
sys.path.append('..')

def fill_padding(data, max_len):
    if len(data) < max_len:
        pad_len = max_len - len(data)
        padding = [0 for _ in range(pad_len)]
        data = data + padding
    else:
        data = data[:max_len]
    return torch.tensor(data)


tags = {
    'positive': 0,
    'negative': 1,
    'neutral': 2,
    '': 3  # 仅占位
}


class TextDataset(Dataset):

    def __init__(self, data: list, tokenizer: BertTokenizer, maxLen: int):
        self.data = data
        self.tokenizer = tokenizer
        self.maxLen = maxLen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        guid = self.data[item]['guid']
        text = self.data[item]['text']
        tag = self.data[item]['tag']
        tag = torch.tensor(tags[tag], dtype=torch.long)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.maxLen,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'guid': guid,
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'tag': tag
        }


def get_text_dataset(path: str):
    with open(path, 'r', encoding='utf-8') as fs:
        data = json.load(fs)

    tokenizer = pretrain_model.getTokenizer()
    return TextDataset(data, tokenizer, config.MAX_LEN)


if __name__ == '__main__':
    data_loader = DataLoader(get_text_dataset(config.TRAIN_DATA_PATH), batch_size=config.BATCH_SIZE, shuffle=True)
    pretrained = pretrain_model.getBert()
    for param in pretrained.parameters():
        param.requires_grad_(False)
    for i, data in enumerate(data_loader):
        print(data)
        out = pretrained(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            token_type_ids=data['token_type_ids']
        )
        print(out['last_hidden_state'].shape)
        break
