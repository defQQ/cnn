import pandas as pd
from torchvision import transforms, datasets
from torch.autograd import Variable
import os
import torch
import torch.nn as nn
import numpy as np


df_test = pd.read_json('data/test.json')
test_dir = 'data/test'

transform_tensor = transforms.Compose([
        transforms.RandomResizedCrop(75),
        transforms.ToTensor(),
    ])

test_datasets = datasets.ImageFolder(os.path.join(test_dir), transform=transform_tensor)
test_loader = torch.utils.data.DataLoader(test_datasets, shuffle=False,  num_workers=0)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(16,2),
            nn.Softmax(),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out


cnn = CNN()
cnn.cuda()

cnn.load_state_dict(torch.load('model'))
cnn.eval()

probabilities = []
for i, (images,_) in enumerate(test_loader):
    images = Variable(images.cuda())
    output = cnn(images)
    preds = output.data.cpu().numpy()

    probabilities.append(preds[0][1])


submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': np.array(probabilities)})
print(submission.head(10))

submission.to_csv('save_result.csv', index=False)





