import os
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torch.utils.data as Data
import torchvision
from PIL import Image
import pandas
import numpy
import matplotlib.pyplot
import torch.optim as optim
import math

print(torch.__version__)
print(torchvision.__version__)

symbolsMap = pandas.read_csv('./data/csv/symbolsMap.csv')
classesMap = {}
for row in symbolsMap.itertuples(index=True):
    classesMap[getattr(row, 'Symbols')] = getattr(row, 'Index')
# end
testResults, trainResults = [], []
preprocessInput = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()])
"""
Training dataset loader
"""


class trainDataset(Dataset):
    def __init__(self, trainSetPath):
        data_info = pandas.read_csv(trainSetPath)
        self.images = numpy.array(data_info.iloc[:, 0], dtype=str)
        labels = numpy.array(data_info.iloc[:, 2])
        self.labels = numpy.array(labels)
    # end

    def __getitem__(self, index):
        imagePath = self.images[index]
        label = self.labels[index]
        return imagePath, label
    # end

    def __len__(self):
        return len(self.images)
# end


"""
Testset data loader
"""


class testDataset(Dataset):
    def __init__(self, trainSetPath):
        data_info = pandas.read_csv(trainSetPath)
        self.images = numpy.array(data_info.iloc[:, 0], dtype=str)
        labels = numpy.array(data_info.iloc[:, 2])
        self.labels = numpy.array(labels)
    # end

    def __getitem__(self, index):
        imagePath = self.images[index]
        label = self.labels[index]
        return imagePath, label
    # end

    def __len__(self):
        return len(self.images)
# end


"""
Convolutional Neural Network
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.prelu1 = torch.nn.PReLU()
        self.prelu2 = torch.nn.PReLU()
        self.tanh = torch.nn.Tanh()
        self.fc1 = torch.nn.Linear(12544, 1024)
        self.fc2 = torch.nn.Linear(1024, 369)
    # end

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=(2, 2))
        x = torch.nn.functional.dropout(x, p=0.25, training=self.training)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.tanh(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)
    # end

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    # end


net = Net().cuda()

print("network", net)


def getLabel(label):
    label = str(label[0])
    index = classesMap.get(label)
    return torch.FloatTensor([index]).long()


# SGD optimizer
optimizer = torch.optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)
# Apdaptive learning scheduler
ap_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# training data
trainData = trainDataset('./data/csv/train.csv')
# load the training data
trainLoader = Data.DataLoader(trainData, batch_size=128, shuffle=True)
digitsData = testDataset('./data/csv/digits-test.csv')
# load test data
testLoader = Data.DataLoader(digitsData, batch_size=128, shuffle=True)

print("done loading sets")

print("len 1", trainLoader.__len__())
print("len 2", testLoader.__len__())

"""Implement training phase"""


def train(optimizer, scheduler):
    net.train()
    scheduler.step()
    for n, sample in enumerate(trainLoader):
        imagePath, label = sample
        image = preprocessInput(Image.open(imagePath[0]))
        image = torch.autograd.Variable(
            image, requires_grad=False, volatile=False).cuda()
        label = torch.autograd.Variable(
            getLabel(label), requires_grad=False, volatile=False).cuda()
        image = image.unsqueeze(0)
        optimizer.zero_grad()
        estimate = net(image)
        loss = torch.nn.functional.nll_loss(input=estimate, target=label)
        loss.backward()
        optimizer.step()
    # end
# end


"""Implement testing phase""""


def test(optimizer):
    net.eval()
    train, test = 0, 0
    for n, sample in enumerate(trainLoader):
        imagePath, label = sample
        image = preprocessInput(Image.open(imagePath[0]))
        image = torch.autograd.Variable(
            image, requires_grad=False, volatile=False).cuda()
        label = torch.autograd.Variable(
            getLabel(label), requires_grad=False, volatile=False).cuda()
        image = image.unsqueeze(0)
        estimate = net(image)
        train = train + \
            estimate.data.max(dim=1, keepdim=False)[1].eq(label.data).sum()
    # end
    for n, sample in enumerate(testLoader):
        imagePath, label = sample
        image = preprocessInput(Image.open(imagePath[0]))
        image = torch.autograd.Variable(
            image, requires_grad=False, volatile=False).cuda()
        label = torch.autograd.Variable(
            getLabel(label), requires_grad=False, volatile=False).cuda()
        image = image.unsqueeze(0)
        estimate = net(image)
        test = test + estimate.data.max(dim=1,
                                        keepdim=False)[1].eq(label.data).sum()
    # end
    trainResults.append(100.0 * train / trainLoader.__len__())
    testResults.append(100.0 * test / testLoader.__len__())

    print('')
    print('train: ' + str(train) + '/' + str(trainLoader.__len__()) +
          ' (' + str(trainResults[-1]) + '%)')
    print('validation: ' + str(test) + '/' +
          str(testLoader.__len__()) + ' (' + str(testResults[-1]) + '%)')
    print('')
# end


for i in range(100):
    print("epoch :", i)
    print("training start")
    train(optimizer, ap_lr_scheduler)
    print("testing start")
    test(optimizer)
# end

matplotlib.pyplot.figure(figsize=(4.0, 5.0), dpi=150.0)
matplotlib.pyplot.plot(trainResults)
matplotlib.pyplot.plot(testResults)
matplotlib.pyplot.show()
