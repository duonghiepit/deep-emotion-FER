from __future__ import print_function
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loaders import PlainDataset, eval_data_dataloader  # Import your custom dataset loader
from deep_emotion import Deep_Emotion, ModelManage  # Import the Deep_Emotion model and ModelManage class
from generate_data import GenerateData  # Import the data generation script

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Your device: {device}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-s', '--setup', type=bool, help='setup the dataset for the first time')
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='data folder that contains data files that downloaded from kaggle (train.csv and test.csv)')
    parser.add_argument('-hparams', '--hyperparams', type=bool,
                        help='True when changing the hyperparameters e.g (batch size, LR, num. of epochs)')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, help='value of learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, help='training/validation batch size')
    parser.add_argument('-t', '--train', type=bool, help='True when training')
    args = parser.parse_args()

    if args.setup:
        generate_dataset = GenerateData(args.data)
        generate_dataset.split_test()
        generate_dataset.save_images('train')
        generate_dataset.save_images('test')
        generate_dataset.save_images('val')

    # Set hyperparameters
    if args.hyperparams:
        epochs = args.epochs
        lr = args.learning_rate
        batchsize = args.batch_size
    else:
        epochs = 100
        lr = 0.005
        batchsize = 128

    if args.train:
        net = Deep_Emotion()
        print("Model architecture: ", net)

        traincsv_file = f"{args.data}/train.csv"
        validationcsv_file = f"{args.data}/val.csv"
        train_img_dir = f"{args.data}/train/"
        validation_img_dir = f"{args.data}/val/"

        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = PlainDataset(csv_file=traincsv_file, img_dir=train_img_dir, data_type='train', transform=transformation)
        validation_dataset = PlainDataset(csv_file=validationcsv_file, img_dir=validation_img_dir, data_type='val', transform=transformation)
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
        val_loader = DataLoader(validation_dataset, batch_size=batchsize, shuffle=False, num_workers=0)

        # Initialize the ModelManage class and start training
        model_manager = ModelManage(net, train_loader, val_loader, lr, device)
        model_manager.train(num_epochs=epochs, save_dir='./models')
