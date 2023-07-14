# coding = utf-8
# -*- coding:utf-8 -*-
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
import utils.config as config

import model.pretrain_model as pretrain_model


from utils.img_util import getImgDataset
from utils.run_util import train, test, predict, device

config.setup_seed()


class ImgModel(nn.Module):
    def __init__(self, fine_tune: bool):
        super().__init__()

        self.vit = pretrain_model.getViT()
        for param in self.vit.parameters():
            param.requires_grad_(fine_tune)

        self.dp = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 3)

    def forward(self, data):
        pixel_values = data['img']['pixel_values'][:, 0].to(device)

        out = self.vit(
            pixel_values=pixel_values
        )
        out = self.fc(self.dp(out['pooler_output']))

        return out


def img_run():
    model = ImgModel(fine_tune=config.FINE_TUNE)
    model.to(device)

    vit_params = list(map(id, model.vit.parameters()))
    down_params = filter(lambda p: id(p) not in vit_params, model.parameters())
    optimizer = AdamW([
        {'params': model.vit.parameters(), 'lr': config.VIT_LR},
        {'params': down_params, 'lr': config.LR}
    ])

    dataset = getImgDataset(config.TRAIN_DATA_PATH)
    # print(len(dataset))
    train_dataset = Subset(dataset, range(0, 3500))
    val_dataset = Subset(dataset, range(3500, 4000))
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    train(model, optimizer, train_loader, val_loader)


def img_test():
    model = torch.load(config.CACHE_MODEL_PATH, map_location=device)
    dataset = getImgDataset(config.TRAIN_DATA_PATH)
    val_dataset = Subset(dataset, range(3500, 4000))
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    print('final validation accuracy:', test(model, val_loader))


def img_pred():
    model = torch.load(config.CACHE_MODEL_PATH, map_location=device)
    test_loader = DataLoader(getImgDataset(config.TEST_DATA_PATH), batch_size=config.BATCH_SIZE, shuffle=False)
    predict(model, test_loader)


if __name__ == '__main__':
    run()
    img_test()
    img_pred()
