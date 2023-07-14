# coding = utf-8
# -*- coding:utf-8 -*-
import json, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor
import config
import model.pretrain_model as pretrain_model

config.setup_seed()

tag_dict = {
    'positive': 0,
    'negative': 1,
    'neutral': 2,
    '': 3  # Placeholder only
}

class ImageDataset(Dataset):
    def __init__(self, data: list, extractor: ViTFeatureExtractor):
        self.data = data
        self.extractor = extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data[index]
        guid = sample['guid']
        image_path = sample['img']
        tag = sample['tag']
        tag = torch.tensor(tag_dict[tag], dtype=torch.long)
        image = self.extractor(
            images=Image.open(config.raw_data_path + image_path),
            return_tensors='pt'
        )

        return {
            'guid': guid,
            'image': image,
            'tag': tag
        }

def get_img_dataset(path: str):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    extractor = pretrain_model.getExtractor()
    return ImageDataset(data, extractor)

if __name__ == '__main__':
    dataset = get_img_dataset(config.TRAIN_DATA_PATH)
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    pretrained_model = pretrain_model.getViT()
    for param in pretrained_model.parameters():
        param.requires_grad_(False)
    for i, data in enumerate(data_loader):
        print(data['image']['pixel_values'].shape)
        output = pretrained_model(
            pixel_values=data['image']['pixel_values'][:, 0]
        )
        print(output['last_hidden_state'].shape)
        print(output['pooler_output'].shape)
        break