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
import gpuProfiler
import torchvision

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
    '''
        Modifications:
            * I removed the BatchNorm2d from conv2 and placed it after adding the residual network. Previously, Batch normalization was being applied after conv2 and downsampling, but before adding the residual. This seemed 
                like it was unnecessary, and the statistics might be not as good?
            * I removed the biases from the convolutional layers since they will be applied by the batch norm.
            * LeakyReLU
            * Did dropout 
    '''

    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()

        # With convolutional layers, Batch normalization is applied to each channel. This means that there are separate (mean, std) parameters for each 
        # channel. The idea is that the same feature map is convolved with the input to produce a single channel, so the outputs in a channel should have 
        # similar information and thus similar statistics.
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU())
        # The second sequence does not include an activation at the end. This is because we add the residual 
        # to the output of the second sequence, and then apply the activation. 
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False))
        self.batchNorm2d = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(p = 0.1)
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.batchNorm2d(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out
        
class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes = 10):
            super(ResNet, self).__init__()
            self.inchannels = 64
            # The input is 224 x 224. I don't know how they get the correct size since
            # Hout = (Hin + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
            # and in this case is (224 + 2*3 - 1*(7 - 1) - 1)/2 + 1 = 112.5
            self.conv1 = nn.Sequential(
                            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
                            nn.BatchNorm2d(64),
                            nn.LeakyReLU())
            # 112 x 112 x 64 goes into maxpool to give us 56 x 56 x 64  
            self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            # Input is 56 x 56 x 64, and output is 56 x 56 x 64
            self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)

            # Input is 56 x 56 x 64 and output is 28 x 28 x 128
            self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
            # Input is 28 x 28 x 128 and output is 14 x 14 x 256
            self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
            # Input is 14 x 14 x 256 and output is 7 x 7 x 512
            self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
            # Input is 7 x 7 x 512 and output is 1x1x512
            self.avgpool = nn.AvgPool2d(7, stride=1)
            # Input is 1 x 512 and output is 1 x 10
            self.fc = nn.Linear(512, num_classes)

        def _make_layer(self, block, channels, blocks, stride=1):
            downsample = None

            if stride != 1 or self.inchannels != channels:
                downsample = nn.Conv2d(self.inchannels, channels, kernel_size=1, stride=stride, bias = False)

            layers = []
            layers.append(block(self.inchannels, channels, stride, downsample))
            self.inchannels = channels
            for i in range(1, blocks):
                layers.append(block(self.inchannels, channels))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, dim=1) # Get the index of the max log-probability
    correct = (predicted == labels).sum().item() # Count correct predictions
    total = labels.size(0) # Total number of predictions
    accuracy = correct / total
    return accuracy

def trainModel(train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader, model: ResNet, num_train_batches: int, enable_tensorbaord: bool = True):
    
    print("training model")

    total_step = len(train_loader)

    # Sets model in training mode
    model.train()

    #Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs)


    print("starting training with num_epochs: ", num_epochs)
    for epoch in range(num_epochs):
        # iterates over epochs
        for i, (images, labels) in enumerate(train_loader):  
            #Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            global_step = epoch * num_train_batches + i


            #Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            accuracy = calculate_accuracy(outputs,labels)

            #Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            gc.collect()
            
            profiler.takeSnapshot()

            if(enable_tensorbaord):
                img_grid = torchvision.utils.make_grid(images)
                # writer.add_image("images",img_grid, global_step=global_step)
                # writer.add_histogram("fc",model.fc.weight, global_step=global_step)
                writer.add_scalar('training loss', loss.item(), global_step)
                writer.add_scalar('training accuracy', accuracy, global_step)
            
            del images, labels, outputs
            print('Batch [{}/{}], Loss {}, Accuracy {}'.format(i,num_train_batches,loss.item(), accuracy))

        scheduler.step()
        print ('Epoch [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, num_epochs, loss.item()))

    model.eval()

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

        print('Accuracy of the network on the {} validation images: {} %'.format(total, 100 * correct / total))


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

    writer = SummaryWriter("resnet/runs/tensorboard")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using cuda: ", torch.cuda.is_available())


    batch_size = 120
 

    train_loader, valid_loader = data_loader(data_dir='./datasets/cifar10',
                                             batch_size=batch_size)

    test_loader = data_loader(data_dir='./datasets/cifar10',
                                  batch_size=batch_size,
                                  test=True)
    
    # If a function is nested inside of another function, the inner function can access variables from the
    # outer function's scope.
    num_classes = 10
    # num_epochs = 20
    num_epochs = 10
    learning_rate = 0.01

    num_train_batches = len(train_loader)
    print("num_train_batches", num_train_batches)

    profiler = gpuProfiler.GpuProfiler(20, "/home/artemis/DNN/resnet/profile", True)

    # The resnet model is broken up into 4 main blocks where each block decreases the size by a factor of two 
    # and increases the number of channels by a factor of two.
    numBlocks = [3, 4, 6, 3]
    model = ResNet(ResidualBlock, numBlocks).to(device)

    if os.path.exists(modelPath):
        model.load_state_dict(torch.load(modelPath))

    writer.add_graph(model, next(iter(train_loader))[0].to(device))

    trainModel(train_loader, valid_loader, model, num_train_batches)

    torch.save(model.state_dict(), modelPath)


    testModel(test_loader,model)

    profiler.generateHtmlPlotFromSnapshot()


    # Tensorboard embedding projector: https://www.youtube.com/watch?v=RLqsxWaQdHE&ab_channel=AladdinPersson
    # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html



