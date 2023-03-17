import os
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json

# Define inference harware

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load latest model

model_dir = 'models/resnet152'
available_models = []

for model in os.listdir(model_dir):
  if os.path.isfile(os.path.join(model_dir, model)) and model.endswith(".pt"):
    available_models.append(model)

available_models.sort()
model_path = os.path.join(model_dir, available_models[-1])
model_uuid = available_models[-1].split("-")[2][:-3]

print("Using model: ", model_path)

model = models.resnet152()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 8)
model.load_state_dict(torch.load(model_path))
model.to(device)

# Instanciate input

test_df = pd.read_csv('data/processed/test.csv')
images = {}
for i in range(len(test_df)):
  images[str(test_df.iloc[i]['idx_test'])] = test_df.iloc[i]['path_img']

test_transform = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Perform

model.eval()

results = {}
with torch.no_grad():
  for idx, path in images.items():
    img = Image.open('data/processed/'+path).convert('RGBA').convert('RGB')
    img_tensor = test_transform(img).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    output = model(img_tensor)
    _, predicted = torch.max(output.data, 1)
    results[idx] = predicted.item()

tojson = {
  "target": results
}

with open(f'results/f1-{model_uuid}.json', 'w') as f:
  f.write(json.dumps(tojson))
