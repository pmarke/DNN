import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import cv2 as cv
from model import YoloV11
import os

ImageSize=(32,31)

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
                transforms.Resize(ImageSize),
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



def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, dim=1) # Get the index of the max log-probability
    correct = (predicted == labels).sum().item() # Count correct predictions
    total = labels.size(0) # Total number of predictions
    accuracy = correct / total
    return accuracy

def trainModel(train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader, model: YoloV11, num_train_batches: int):
    
    print("training model")

    total_step = len(train_loader)

    # Sets model in training mode
    model.train()

    #Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs)

    # Use automatic mixed precision for training
    # GradScaler helps to prevent underflow in gradients when using float16
    # This is useful for training large models on GPUs with limited memory.
    # https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    scaler = torch.amp.GradScaler(device.type)

    print("starting training with num_epochs: ", num_epochs)
    for epoch in range(num_epochs):
        # iterates over epochs
        for i, (images, labels) in enumerate(train_loader):  
            #Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Runs the forward pass under ``autocast``. This will allow it to switch between 
            # float16 and float32 as needed.
            # This is useful for training large models on GPUs with limited memory. It is necessary 
            # for flash attention
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                #Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            accuracy = calculate_accuracy(outputs,labels)

            #Backward and optimize
            optimizer.zero_grad()
            # When using automatic mixed precision, we scale the loss to prevent underflow since 
            # we are using mixed precision in training.
            # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()       
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()
            
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
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} validation images: {} %'.format(total, 100 * correct / total))


def testModel(test_loader: torch.utils.data.DataLoader, model: YoloV11):
     model.eval()
     with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))

if __name__ =='__main__':

    modelPath = "/home/artemis/DNN/yolo11/model.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using cuda: ", torch.cuda.is_available())


    batch_size = 150
 

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
    learning_rate = 0.001

    num_train_batches = len(train_loader)
    print("num_train_batches", num_train_batches)

    # The resnet model is broken up into 4 main blocks where each block decreases the size by a factor of two 
    # and increases the number of channels by a factor of two.
    model = YoloV11(dropout=0.1).to(device)

    if os.path.exists(modelPath):
        model.load_state_dict(torch.load(modelPath))


    trainModel(train_loader, valid_loader, model, num_train_batches)

    torch.save(model.state_dict(), modelPath)

    testModel(test_loader,model)

    # Tensorboard embedding projector: https://www.youtube.com/watch?v=RLqsxWaQdHE&ab_channel=AladdinPersson
    # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html



