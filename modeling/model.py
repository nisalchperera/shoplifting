import os
import torch

from tqdm import tqdm
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module, LSTM, Linear, CrossEntropyLoss



class ShopliftingClassifier(Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes) -> None:
        super(ShopliftingClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class ShopliftingModel(Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, yolo_version=8, yolo_size="m") -> None:
        super(ShopliftingModel, self).__init__()

        self.pose = YOLO(f"models/yolov{yolo_version}{yolo_size}-pose.pt")
        self.classifer = ShopliftingClassifier(input_size, hidden_size, num_layers, num_classes)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    
    def accuracy(self, y_pred, y_true):
        _, predicted = torch.max(y_pred.data, 1)
        return (predicted == y_true).sum().item()
    
    def train(self, train_loader, criterion, optimizer, e=None, writer=None):
        
        self.classifer.train()

        losses = 0.0
        acc = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self(images)
            loss = criterion(outputs, labels)
            losses = losses + loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        # print("")
        # os.makedirs(save_dir, exist_ok=True)

        # torch.save(self, os.path.join(save_dir, 'shoplifting_detector.pth'))
        
    def train_epochs(self, epochs, dataset, save_dir="models/"):
        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters())
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        writer = SummaryWriter()

        train_pbar = tqdm(range(epochs), desc="Model Training")
        for e in train_pbar:
            loss, acc = self.train(train_loader=dataloader, criterion=criterion, optimizer=optimizer)

            train_pbar.set_postfix_str(f"Epoch: {e}, Accuracy: {acc}, Loss: {loss}")

            # accuracy = 100 * total_correct / total_samples
            # avg_loss = running_loss / 100
            writer.add_scalar('Training Loss', loss, e)
            writer.add_scalar('Training Accuracy', acc, e)

        os.makedirs(save_dir, exist_ok=True)
        torch.save(self, os.path.join(save_dir, 'shoplifting_detector.pth'))
