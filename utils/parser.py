import argparse
import config

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='img_and_text', help='Data type to be used: img_only, text_only, img_and_text')
    parser.add_argument('--train', action='store_true', help='Train')
    parser.add_argument('--test', action='store_true', help='Test on the validation set')
    parser.add_argument('--predict', action='store_true', help='Generate predictions for the test set')

    parser.add_argument('--train_with_label_path', type=str, default=config.train_with_label_path, help='Path to train.txt')
    parser.add_argument('--test_without_label_path', type=str, default=config.test_without_label_path, help='Path to test_without_label.txt')
    parser.add_argument('--raw_data_path', type=str, default=config.raw_data_path, help='Path to image and text data')
    parser.add_argument('--train_data_path', type=str, default=config.train_data_path, help='Path to preprocessed training and validation set')
    parser.add_argument('--test_data_path', type=str, default=config.test_data_path, help='Path to preprocessed test set')
    parser.add_argument('--cache_model_path', type=str, default=config.cache_model_path, help='Path to save trained models')
    parser.add_argument('--prediction_path', type=str, default=config.prediction_path, help='Path to save test set predictions')

    parser.add_argument('--seed', type=int, default=config.seed, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='Batch size')
    parser.add_argument('--epoch', type=int, default=config.epoch, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=config.lr, help='Learning rate for downstream tasks')

    parser.add_argument('--bert', type=str, default=config.text_pretrained_model_name_or_path, help='Path to BERT (bert-base-multilingual-cased)')
    parser.add_argument('--bert_lr', type=float, default=config.bert_lr, help='Learning rate for BERT fine-tuning')

    parser.add_argument('--vit', type=str, default=config.img_pretrained_model_name_or_path, help='Path to ViT (vit-base-patch16-224-in21k)')
    parser.add_argument('--vit_lr', type=str, default=config.vit_lr, help='Learning rate for ViT fine-tuning')

    return parser.parse_args()
