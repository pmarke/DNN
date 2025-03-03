import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import cv2 as cv
import gc
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def data_loader(data_dir,
                    batch_size,
                    random_seed=42,
                    valid_size=0.1,
                    shuffle=True,
                    test=False):

        # The means and standard deviations of the CIFAR-10 dataset for each channel. 
        # These values are used to normalize the dataset, and have been previously calculated.
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

        # define transforms
        transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
        ])

        if test:
            dataset = datasets.CIFAR10(
              root=data_dir, train=False,
              download=True, transform=transform,
            )

            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle
            )

            return data_loader

        # load the dataset
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=transform,
        )

        valid_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=transform,
        )

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler)

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler)

        return (train_loader, valid_loader)

class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                            nn.BatchNorm2d(out_channels),
                            nn.LeakyReLU())
            # The second sequence does not include an activation at the end. This is because we add the residual 
            # to the output of the second sequence, and then apply the activation. 
            self.conv2 = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                            nn.BatchNorm2d(out_channels))
            self.downsample = downsample
            self.relu = nn.LeakyReLU()
            self.out_channels = out_channels

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.conv2(out)
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out
        
class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes = 10):
            super(ResNet, self).__init__()
            self.inchannels = 64
            # The input is 224 x 224. I don't know how they get the correct size since
            # Hout = (Hin + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
            # and in this case is (224 + 2*3 - 1*(7 - 1) - 1)/2 + 1 = 112.5
            self.conv1 = nn.Sequential(
                            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                            nn.BatchNorm2d(64),
                            nn.LeakyReLU())
            # 112 x 112 x 64 goes into maxpool to give us 56 x 56 x 64  
            self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
            self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
            self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
            self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512, num_classes)

        def _make_layer(self, block, channels, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inchannels != channels:

                downsample = nn.Sequential(
                    nn.Conv2d(self.inchannels, channels, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(channels),
                )
            layers = []
            layers.append(block(self.inchannels, channels, stride, downsample))
            self.inchannels = channels
            for i in range(1, blocks):
                layers.append(block(self.inchannels, channels))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            print("first shape {}".format(x.shape))
            x = self.maxpool(x)
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x



def trainModel(train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader, model: ResNet):
    
    print("training model")

    total_step = len(train_loader)

    # Sets model in training mode
    model.train()

    #Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  


    print("starting training with num_epochs: ", num_epochs)
    for epoch in range(num_epochs):
        # iterates over epochs
        for i, (images, labels) in enumerate(train_loader):  
            #Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            #Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            #Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()
            
            print('Batch {}, Loss {}'.format(i,loss.item()))

        print ('Epoch [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, num_epochs, loss.item()))

    #Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))


def testModel(test_loader: torch.utils.data.DataLoader, model: ResNet):
     model.eval()
     with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))

if __name__ =='__main__':

    modelPath = "/home/artemis/DNN/resnet/model.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using cuda: ", torch.cuda.is_available())

    train_loader, valid_loader = data_loader(data_dir='./datasets/cifar10',
                                             batch_size=32)

    test_loader = data_loader(data_dir='./datasets/cifar10',
                                  batch_size=32,
                                  test=True)
    
    # If a function is nested inside of another function, the inner function can access variables from the
    # outer function's scope.
    num_classes = 10
    # num_epochs = 20
    num_epochs = 1
    batch_size = 16
    learning_rate = 0.01

    # The resnet model is broken up into 4 main blocks where each block decreases the size by a factor of two 
    # and increases the number of channels by a factor of two.
    numBlocks = [3, 4, 6, 3]
    model = ResNet(ResidualBlock, numBlocks).to(device)

    if os.path.exists(modelPath):
        model.load_state_dict(torch.load(modelPath))

    trainModel(train_loader, valid_loader, model)

    torch.save(model.state_dict(), modelPath)

    # Sets the model in evaluation mode
    # model.eval()

    testModel(test_loader,model)



