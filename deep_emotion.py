import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

def conv_block(input_channel, output_channel, kernel_size, stride=1, padding=0):
    conv_block = nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size),
        nn.ReLU(),
        nn.Conv2d(output_channel, output_channel, kernel_size),
        nn.MaxPool2d(2, 2),
        nn.ReLU()
    )

    return conv_block

class Deep_Emotion(nn.Module):
    def __init__(self):
        super(Deep_Emotion, self).__init__()
        self.conv1 = conv_block(1, 10, 3)
        self.conv2 = conv_block(10, 10, 3)
        self.norm = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(810, 50)
        self.fc2 = nn.Linear(50, 7)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3*2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)  # Set align_corners to True
        x = F.grid_sample(x, grid, align_corners=True)  # Set align_corners to True

        return x
    
    def forward(self, input):
        out = self.stn(input)

        out = self.conv1(out)
        out = self.conv2(out)

        out = F.dropout(out)
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out
    
class ModelManage:
    def __init__(self, model, train_loader, val_loader, lr, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = device

    def train(self, num_epochs, save_dir='.'):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'best-{self.model.__class__.__name__}.pth')
        print("============================================Start Training============================================")
        for epoch in range(num_epochs):
            start_time = time.time()
            self.model.train()  # Set the model to training mode
            train_loss = 0
            train_correct = 0

            # Train the model
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                train_loss += loss.item()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                train_correct += torch.sum(preds == labels.data)

            # Validate the model
            val_correct, validation_loss = self.evaluate(self.val_loader)

            train_loss /= len(self.train_loader.dataset)
            train_acc = train_correct.double() / len(self.train_loader.dataset)
            validation_loss /= len(self.val_loader.dataset)
            val_acc = val_correct.double() / len(self.val_loader.dataset)

            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'time: {int(time.time() - start_time)}s, '
                  f'loss: {train_loss:.4f}, '
                  f'val_loss: {validation_loss:.4f}, '
                  f'train_acc: {train_acc * 100:.3f}%, '
                  f'val_acc: {val_acc * 100:.3f}%')

        torch.save(self.model.state_dict(), save_path)
        print("============================================Training Finished============================================")
    
    def evaluate(self, loader):
        self.model.eval()
        val_correct = 0
        validation_loss = 0
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.to(self.device), labels.to(self.device)
                val_outputs = self.model(data)
                val_loss = self.criterion(val_outputs, labels)
                validation_loss += val_loss.item()
                _, val_preds = torch.max(val_outputs, 1)
                val_correct += torch.sum(val_preds == labels.data)
        return val_correct, validation_loss
        