import os
import torch

import torch.nn.functional as F

from glob import glob

from tqdm import tqdm
from ultralytics import YOLO

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module, Conv3d, BatchNorm3d, ReLU, MaxPool3d, AdaptiveAvgPool3d, Sequential, Identity, Linear, ModuleList, CrossEntropyLoss, Flatten, Sigmoid, BCEWithLogitsLoss

    
class ShopliftingDetector(Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Initial 3D conv layer
        self.conv1 = Conv3d(3, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3))
        self.bn1 = BatchNorm3d(64)
        self.relu = ReLU()
        self.maxpool = MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        
        # (2+1)D ResNet blocks
        self.layers = ModuleList([
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
        ])

        # Classification head
        self.classifier = ModuleList([
            AdaptiveAvgPool3d((1,1,1)),
            Linear(512, num_classes),  # 2 classes: normal, shoplifting
            Sigmoid()
        ])
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for module in self.layers:
            x = module(x)

        for module in self.classifier:
            x = module(x)
        
        return x

class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Spatial convolution
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=(1,3,3), 
                               stride=(1,stride,stride), padding=(0,1,1))
        self.bn1 = BatchNorm3d(out_channels)
        
        # Temporal convolution 
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=(3,1,1),
                               stride=(1,1,1), padding=(1,0,0))
        self.bn2 = BatchNorm3d(out_channels)
        
        self.relu = ReLU()
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Sequential(
                Conv3d(in_channels, out_channels, kernel_size=1, stride=(1,stride,stride)),
                BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = Identity()
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out
    

class Model():
    def __init__(self, **kwargs) -> None:
        
        device = kwargs.pop("device", None)
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = ShopliftingDetector(**kwargs).to(self.device)

    def accuracy(self, y_pred, y_true):
        # Get the predicted class (index of maximum value)
        predicted = torch.argmax(y_pred, dim=1)
        
        # Get the true class (index of the 1 in one-hot encoding)
        true_class = torch.argmax(y_true, dim=1)
        
        # Compare predictions with true classes
        correct = (predicted == true_class)
        
        # Calculate accuracy
        return correct.float().mean().item()
    
    def train(self, train_loader, criterion, optimizer, e=None, writer=None):
        self.model.train()

        losses = 0.0
        acc = 0.0
        total_samples = 0.0

        for batch_idx, (keypoints, labels) in enumerate(train_loader):
            keypoints = keypoints.to(self.device).float()
            labels = F.one_hot(labels.to(self.device), num_classes=2).float()
            
            outputs = self.model(keypoints)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses += loss.item()
            total_samples += labels.size(0)
            acc += self.accuracy(outputs, labels)

            if writer and e:
                accuracy = 100 * acc / total_samples
                avg_loss = losses / 100
                writer.add_scalar('Batch Training Loss', avg_loss, e * len(train_loader) + batch_idx)
                writer.add_scalar('Batch Training Accuracy', accuracy, e * len(train_loader) + batch_idx)
                losses = 0.0
                acc = 0
                total_samples = 0
            
        return losses / len(train_loader), acc / len(train_loader)

    def validate(self, val_loader, criterion, epoch, writer):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        running_loss = 0.0
        
        with torch.no_grad():
            for keypoints, labels in val_loader:
                keypoints = keypoints.to(self.device).float()
                labels = F.one_hot(labels.to(self.device), num_classes=2).float()
                
                outputs = self.model(keypoints)
                loss = criterion(outputs, labels)
                
                total_samples += labels.size(0)
                total_correct += self.accuracy(outputs, labels) * labels.size(0)
                running_loss += loss.item()
        
        accuracy = 100 * total_correct / total_samples
        avg_loss = running_loss / len(val_loader)
        writer.add_scalar('Validation Loss', avg_loss, epoch)
        writer.add_scalar('Validation Accuracy', accuracy, epoch)
        
        return accuracy, avg_loss
    
    def logging_dir(self, log_dir=None, exist_ok=False):
        if not log_dir:
            log_dir = "runs"
            dirs = list(glob(os.path.join(log_dir, "train*")))
            if len(dirs):
                log_dir = os.path.join(log_dir, f"train{len(dirs)}")
            else:
                log_dir = os.path.join(log_dir, f"train")
            os.makedirs(log_dir, exist_ok=False)
            return log_dir
        else:
            os.makedirs(log_dir, exist_ok=exist_ok)
            return log_dir
        
    def train_epochs(self, epochs, train_dataset, eval_dataset=None, eval_every=0,save_dir="models/"):
        criterion = BCEWithLogitsLoss() # CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), 0.0001)
        dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        if eval_dataset:
            eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=True)


        writer = SummaryWriter(log_dir=self.logging_dir())

        train_pbar = tqdm(range(epochs), desc="Model Training")
        for e in train_pbar:
            loss, acc = self.train(train_loader=dataloader, criterion=criterion, optimizer=optimizer)

            if eval_every and e % eval_every:
                eval_acc, eval_loss = self.validate(eval_dataloader, criterion, e, writer)
                train_pbar.set_postfix_str(f"T Acc: {round(acc, 2)}, T L: {round(loss, 2)}, Eval Acc: {round(eval_acc, 2)}, Eval L: {round(eval_loss, 2)}")
            else:
                train_pbar.set_postfix_str(f"T Acc: {round(acc, 2)}, T L: {round(loss, 2)}")

            # accuracy = 100 * total_correct / total_samples
            # avg_loss = running_loss / 100
            writer.add_scalar('Training Loss', loss, e)
            writer.add_scalar('Training Accuracy', acc, e)

        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model, os.path.join(save_dir, 'shoplifting_detector_3d_cnn.pth'))