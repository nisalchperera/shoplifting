import os
import torch

import torch.nn.functional as F

from glob import glob

from tqdm import tqdm
from ultralytics import YOLO

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module, LSTM, Linear, CrossEntropyLoss, Flatten, Sigmoid, BCEWithLogitsLoss


class ShopliftingModel(Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, yolo_version=8, yolo_size="m") -> None:
        super(ShopliftingModel, self).__init__()

        self.pose = YOLO(f"models/yolov{yolo_version}{yolo_size}-pose.pt")
        self.classifer = ShopliftingClassifier(input_size, hidden_size, num_layers, num_classes)
        

    def forward(self, image):

        with torch.no_grad():
            poses = []

            for img in range(image.shape[1]):
                imgs = image[:, img, :,:,:]

                x = self.pose(imgs, device=self.device)

                ## preprocess x
                x = x[0].keypoints.data

                poses.append(x)

        return self.classifer(x)


class ShopliftingClassifier(Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes) -> None:
        super(ShopliftingClassifier, self).__init__()
        self.num_keypoints = input_size[0] * input_size[1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.flatten = Flatten(start_dim=2)
        
        self.lstm = LSTM(self.num_keypoints, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_keypoints, 2)
        batch_size, _, _, _ = x.size()
        
        # Flatten the input
        x = self.flatten(x)  # Shape: (batch_size, sequence_length, num_keypoints * 2)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Use the output of the last time step
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)
    

class BaseModel():

    def __init__(self, *args, **kwargs) -> None:

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ShopliftingClassifier(**kwargs).to(self.device)

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
        dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

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
        torch.save(self, os.path.join(save_dir, 'shoplifting_detector.pth'))