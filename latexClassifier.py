import os
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as Data
import torchvision
from PIL import Image
import numpy
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.optim as optim
import math
import time

print(torch.__version__)
print(torchvision.__version__)


DigitsImagesFolder = "./digits/"
testAccuracy = []
"""
Image transforms
"""
trainTransform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor()
])

trainInverseTransform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage()
])

"""
Show samples in a grid
img: list of images.
"""


def show(img):
    npimg = img.numpy()
    plt.imshow(numpy.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()
# end


"""
Load the data folder and prepare training set and testing set
dataFolder: directory containing the images
showImage: show sample while preparing training and resting set
ratio: training to test split ratio
shuffle: shuffle example
"""


def getDataLoader(dataFolder=DigitsImagesFolder, showImages=False, ratio=0.25, shuffle=False):
    trainDataset = datasets.ImageFolder(
        root=dataFolder, transform=trainTransform)
    testDataset = datasets.ImageFolder(
        root=dataFolder, transform=trainTransform)
    numTrain = len(trainDataset)
    indices = list(range(numTrain))
    split = int(numpy.floor(ratio * numTrain))
    print("num train - ", numTrain)

    if shuffle:
        numpy.random.shuffle(indices)
    # end

    trainIdx, testIdx = indices[split:], indices[:split]
    trainSampler = SubsetRandomSampler(trainIdx)
    testSampler = SubsetRandomSampler(testIdx)
    trainLoader = torch.utils.data.DataLoader(trainDataset,
                                              batch_size=128, sampler=trainSampler,
                                              num_workers=4)
    testLoader = torch.utils.data.DataLoader(testDataset,
                                             batch_size=128, sampler=testSampler,
                                             num_workers=4)
    if showImages:
        sampleLoader = torch.utils.data.DataLoader(
            trainDataset, batch_size=4, shuffle=shuffle, num_workers=1)
        sampleIter = iter(sampleLoader)
        imageBatch = sampleIter.next()
        print("image type - ", type(imageBatch[0]))
        show(make_grid(imageBatch[0]))
        print('Ground Truth :', ' '.join('%5s' % label.item()
                                         for label in imageBatch[1]))
    # end
    return (trainLoader, testLoader)
# end


"""
Convolution Neural Network
"""


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3)
        self.conv1_bn = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3)
        self.conv2_bn = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3)
        self.conv3_bn = torch.nn.BatchNorm2d(32)
        self.fc1 = torch.nn.Linear(576, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.tanh = nn.Tanh()
        self.prelu = nn.PReLU()
        # end

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = self.prelu(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = x.view(-1, self.num_flat_features(x))
        x = nn.functional.dropout(x, p=0.25, training=self.training)
        x = self.fc1(x)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        x = self.tanh(x)
        return torch.nn.functional.log_softmax(x, dim=1)
        # end

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        # print("size - ", size)
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    # end
# end


"""
Train the model.
model: model
trainSet: training images
device: cuda or cpu
optimizer: Adam
epoch: epoch number
"""


def train(model, trainSet, device, optimizer, epoch):
    model.train()
    for batchId, (tensorInput, tensorTarget) in enumerate(trainSet):
        inputData, label = tensorInput.to(device), tensorTarget.to(device)
        optimizer.zero_grad()
        output = model(inputData)
        trainLoss = nn.functional.nll_loss(output, label)
        trainLoss.backward()
        optimizer.step()

        if batchId % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batchId * len(inputData), len(trainSet.dataset),
                100. * batchId / len(trainSet), trainLoss.item()))
    # end
# end


"""Test the model.
    model : model
    testSet: test data
    device: cuda or cpu
    optimizer: Adam
    epoch: epoch number
"""


def test(model, testSet, device, optimizer, epoch):
    model.eval()
    testLoss = 0
    correctPredictions = 0
    with torch.no_grad():
        for tensorInput, tensorTaget in testSet:
            inputData, label = tensorInput.to(device), tensorTaget.to(device)
            output = model(inputData)
            testLoss = testLoss + \
                nn.functional.nll_loss(
                    output, label, size_average=False).item()
            prediction = output.max(1, keepdim=True)[1]
            correctPredictions = correctPredictions + \
                prediction.eq(label.view_as(prediction)).sum().item()
        # end
    # end
    testLoss = testLoss / len(testSet.dataset)
    print('\nEpoch:{}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, testLoss, correctPredictions, len(testSet.dataset), 100. * correctPredictions / len(testSet.dataset)))
    testAccuracy.append(100. * correctPredictions / len(testSet.dataset))
# end


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network().to(device)
    print("model - ", model)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-08)
    trainSet, testSet = getDataLoader(
        showImages=True, ratio=0.75, shuffle=True)
    print("train set length - ", len(trainSet))
    print("test set length - ", len(testSet))

    for epoch in range(20):
        train(model, trainSet, device, optimizer, epoch)
        test(model, testSet, device, optimizer, epoch)
    # end

    plt.figure(figsize=(4.0, 5.0), dpi=150.0)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy as a function of number of epochs')
    plt.legend(loc='lower right')
    plt.plot(testAccuracy)
    plt.show()


if __name__ == '__main__':
    main()
