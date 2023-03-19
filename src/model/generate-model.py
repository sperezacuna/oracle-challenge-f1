import os
import time
import copy
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def train_model(model, criterion, optimizer, scheduler, num_epochs):
  begin_time = time.time()
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  statistics = {
    "loss" : {
      "training": [],
      "validation": [],
    },
    "accuracy" : {
      "training": [],
      "validation": [],
    }
  }
  for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 11)
    # Each epoch has a training and validation phase
    for phase in ['training', 'validation']:
      if phase == 'training':
        model.train() # Set model to training mode
      else:
        model.eval()  # Set model to evaluate mode
      running_loss = 0.0
      running_corrects = 0
      # Iterate over data.
      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'training'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)
          # backward + optimize only if in training phase
          if phase == 'training':
            loss.backward()
            optimizer.step()
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
      if phase == 'training':
        scheduler.step()
      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]
      statistics["loss"][phase].append(epoch_loss)
      statistics["accuracy"][phase].append(epoch_acc.item())
      print(f'{phase}: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
      if phase == 'validation' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    print()
  time_elapsed = time.time() - begin_time
  print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Final model Acc: {(max(statistics["accuracy"]["validation"]+[0])):4f}')
  # load best model weights
  model.load_state_dict(best_model_wts)
  return model, statistics

# Definition of the dataset class
class FoodDataset(Dataset):
  def __init__(self, csv_file, root_dir, transform=None):
    self.food_frame = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform
      
  def __len__(self):
    return len(self.food_frame)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    path = os.path.join(self.root_dir,
                        self.food_frame.iloc[idx, 1])
    image = Image.open(path).convert("RGBA").convert('RGB')
    label = self.food_frame.iloc[idx, 2]
    if self.transform:
      image = self.transform(image)
    return image, label

# Image transformations
data_transforms = {
  'training': transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing()
  ]),
  'validation': transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]),
}

# Datasets creation
datasets = {
  x: FoodDataset(csv_file='data/processed/'+x+'.csv', root_dir='data/processed', transform=data_transforms[x])
  for x in ['training', 'validation']
}
dataloaders = {
  x: DataLoader(datasets[x], batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
  for x in ['training', 'validation']
}
dataset_sizes = {x: len(datasets[x]) for x in ['training', 'validation']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ",device)
print()

# Download pre-trained model
model = torchvision.models.resnet152(weights='DEFAULT')
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 8.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 8)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.0075, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model, statistics = train_model(model, criterion, optimizer, scheduler, num_epochs=25)

model_uuid = uuid.uuid4().hex
timestamp = datetime.now().strftime("%Y-%m-%d@%H:%M")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Model Statistics')
ax1.plot(statistics['accuracy']['training'])
ax1.plot(statistics['accuracy']['validation'])
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax2.plot(statistics['loss']['training'])
ax2.plot(statistics['loss']['validation'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
fig.legend(['Train accuracy', 'Validation accuracy', 'Train loss', 'Validation loss'])
fig.savefig(f'models/resnet152/foodmodel-[acc:{(max(statistics["accuracy"]["validation"]+[0])):6f}]-{model_uuid}.png')

torch.save(model.state_dict(), f'models/resnet152/foodmodel-[acc:{(max(statistics["accuracy"]["validation"]+[0])):6f}]-{model_uuid}.pt')
