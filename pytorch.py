import numpy as np
import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.autograd import Variable
import os


train_dir = 'data/train'
validation_dir = 'data/val'

transform_tensor = transforms.Compose([
        transforms.RandomResizedCrop(75),
        transforms.ToTensor(),
    ])

train_datasets = datasets.ImageFolder(os.path.join(train_dir), transform=transform_tensor)

val_datasets = datasets.ImageFolder(os.path.join(validation_dir), transform=transform_tensor)

print(len(train_datasets))

num_epochs = 2
batch_size = 32
learning_rate = 0.0003


train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False,  num_workers=4)


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

#cnn.cuda()


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

train_loss_epoch = []
accuracy_epoch = []

# Train the Model
for epoch in range(num_epochs):
    loss_mini = []
    correct_epoch = []
    for i, (images, labels) in enumerate(train_loader):
        print(i)
        total = 0
        correct = 0
        #images = Variable(images).cuda()
        images = Variable(images)
        target = labels
        #labels = Variable(labels).cuda()
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_mini.append(loss.data[0])
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        #correct += (predicted == target.cuda()).sum()
        correct += (predicted == target).sum()
        correct_epoch.append(100*correct/total)

        if (i + 1) % 10 == 0:\
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Accu: %d %%'\
                % (epoch + 1, num_epochs, i + 1, len(train_datasets) // batch_size, loss.data[0],100 * correct / total))

    accuracy_epoch.append((np.array(correct_epoch).mean()))
    train_loss_epoch.append((np.array(loss_mini).mean()))



torch.save(cnn.state_dict(),'model')
