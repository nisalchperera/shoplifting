import os
import torch

import torch.nn.functional as F

from glob import glob

from tqdm import tqdm
from ultralytics import YOLO

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module, Conv3d, BatchNorm3d, ReLU, MaxPool3d, AdaptiveAvgPool3d, Sequential, Identity, Linear, ModuleList, CrossEntropyLoss, Flatten, Sigmoid, BCEWithLogitsLoss

    
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DSimple(nn.Conv3d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv3DSimple(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3DSimple(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ShopliftingDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(ShopliftingDetector, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layers = ModuleList([
            self._make_layer(BasicBlock, 64, 3),
            self._make_layer(BasicBlock, 128, 4, stride=2),
            self._make_layer(BasicBlock, 256, 6, stride=2),
            self._make_layer(BasicBlock, 512, 3, stride=2)
        ])
        # self.layer1 = 
        # self.layer2 = 
        # self.layer3 = 
        # self.layer4 = 

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, 
                        kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=(1, stride, stride), downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    

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

        batch_pbar = tqdm(enumerate(train_loader), desc=f"Epoch {e}", total=len(train_loader))
        for batch_idx, (keypoints, labels) in batch_pbar:
            batch_pbar.set_postfix_str(f"Batch: {batch_idx}")

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
        dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,) # num_workers=os.cpu_count())

        if eval_dataset:
            eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=True)


        writer = SummaryWriter(log_dir=self.logging_dir())

        train_pbar = tqdm(range(epochs), desc="Model Training")
        for e in train_pbar:
            loss, acc = self.train(train_loader=dataloader, criterion=criterion, optimizer=optimizer, e=e)

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