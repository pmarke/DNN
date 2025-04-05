import numpy as np 
import torch 
import torch.nn as nn 
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter
import gc 
import torchvision
import matplotlib.pyplot as plt



class CustomTemperatureDataset(Dataset):
    def __init__(self, dataDir, timeSteps, train=False):
        self.dataDir = dataDir
        self.train = train
        self.timeSteps = timeSteps
        self.data = pd.read_csv(dataDir, parse_dates=['Date'], index_col='Date')
        self.avgData = self.slidingWindowAvg(self.data.values, 30)

        # self.avgData = np.cos(np.pi/500*np.linspace(0,1000,10000)).reshape(-1,1)
        # print(self.avgData.shape)

        self.scaledData, self.dataOffset, self.dataScale = self.scaleMinMax(self.avgData)
        self.inputs, self.lables = self.createDataset(self.scaledData, self.timeSteps)

        indices = np.arange(len(self.inputs))

        # Shuffle the data (optional, for randomness)
        np.random.seed(0)
        np.random.shuffle(indices)

        # Calculate the split index
        split_index = int(0.8 * len(indices))
        # print(indices)

        if self.train:
            self.inputs = self.inputs[indices[:split_index]]
            self.lables = self.lables[indices[:split_index]]
        else :
            self.inputs = self.inputs[indices[split_index:]]
            self.lables = self.lables[indices[split_index:]]

        self.inputs = torch.from_numpy(self.inputs).float().unsqueeze(1)
        print("input shape: ", self.inputs.shape)
        # print(self.inputs[0,:,:])
        self.lables = torch.from_numpy(self.lables).float()

        # self.plotFirst10()

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.lables[idx]
    
    @staticmethod
    def slidingWindowAvg(data: np.array, windowSize : int):

        lpfData = []
        # numWindows = int(len(data)/windowSize)
        # for ii in range(numWindows):
        #     lpfData.append( np.mean(data[ii*windowSize:ii*windowSize+windowSize]))

        for ii in range(len(data)-windowSize):
            lpfData.append(np.mean(data[ii:ii+windowSize]))

        return np.asarray(lpfData).reshape(-1,1)



    def plotFirst10(self):
        '''Plots the first 10 inputs, labels, and the original scaled data'''
        # Plot the first 10 inputs and labels
        fig, axs = plt.subplots(10, 1, figsize=(10, 20))
        for i in range(10):
            axs[i].plot(self.inputs[i*self.timeSteps].squeeze().numpy(), label='Input')
            axs[i].scatter(len(self.inputs[i*self.timeSteps]) - 1, self.lables[i*self.timeSteps].item(), color='red', label='Label')
            start_idx = i
            end_idx = i + self.timeSteps + 1
            # axs[i].plot(range(self.timeSteps + 1), self.scaledData[start_idx:end_idx, 0], color='green', linestyle='--', label='Original Scaled Data')
            axs[i].legend()
            axs[i].set_title(f'Sample {i+1}')
        plt.tight_layout()

        # Plot the entire scaled data in a separate figure
        plt.figure(figsize=(12, 6))
        plt.plot(self.scaledData[:, 0], label='Scaled Data', color='blue')
        plt.title('Entire Scaled Data')
        plt.xlabel('Time')
        plt.ylabel('Scaled Value')
        plt.legend()

        # Plot the entire original data in a separate figure
        plt.figure(figsize=(12, 6))
        plt.plot(self.avgData[:, 0], label='Original Data', color='orange')
        plt.title('Entire Original Data')
        plt.xlabel('Time')
        plt.ylabel('Original Value')
        plt.legend()
        plt.show()



    @staticmethod
    def scaleMinMax(data: np.array):
        ''' Scales the data so that all values are between 0 and 1'''
        offset = data.min()
        scale = data.max() - offset
        data = (data - offset) / scale
        return data, offset, scale

    @staticmethod
    def createDataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            # x = data[i:(i+time_step),0]
            X.append(data[i:(i + time_step), 0]) 
            y.append(data[i + time_step, 0]) 

        X = np.array(X)
        y = np.array(y)
        X.reshape(X.shape[0],X.shape[1])
        return X,y

class TemperatureTcn(nn.Module):
    def __init__(self):
        super(TemperatureTcn, self).__init__()
        
        #input is 100 output is 50
        self.layer1 = nn.Sequential(
                    nn.Conv1d(1, 5, kernel_size = 10, stride = 2, padding = 4, bias = False),
                    nn.BatchNorm1d(5),
                    nn.LeakyReLU())
        
        # input is 50, output is 25
        self.layer2 = nn.Sequential(
                    nn.Conv1d(5, 10, kernel_size = 10, stride = 2, padding = 4, bias = False),
                    nn.BatchNorm1d(10),
                    nn.LeakyReLU())
        
        # input is 25, output is 13
        self.layer3 = nn.Sequential(
                    nn.Conv1d(10, 5, kernel_size = 5, stride = 2, padding = 2, bias = False),
                    nn.BatchNorm1d(5),
                    nn.LeakyReLU())
        
        # input is 12, output is 7

        self.layer4 = nn.Sequential(
                    nn.Conv1d(5, 1, kernel_size = 5, stride = 2, padding = 2, bias = False),
                    nn.BatchNorm1d(1),
                    nn.LeakyReLU())
        
        self.fc = nn.Linear(7,1)
        self.sig = nn.Sigmoid()

    def forward(self,x):
        # print("i:",x.shape)
        x = self.layer1(x)
        # print("o1", x.shape)
        x = self.layer2(x)
        # print("o2", x.shape)
        x = self.layer3(x)
        # print("o3", x.shape)
        x = self.layer4(x)
        # print("o4", x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        # print("fc", x.shape)
        x = self.sig(x)
        return x.squeeze()


def trainModel(trainLoader: DataLoader, model : nn.Module, numEpochs, learningRate: float, enableTensorboard: bool):
    
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, weight_decay = 0.001, momentum = 0.9)  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = numEpochs/5)


    numBatches = len(trainLoader)

    print("starting training with num_epochs: ", numEpochs)
    for epoch in range(numEpochs):
        # iterates over epochs
        for i, (inputs, labels) in enumerate(trainLoader):  
            # print(inputs.shape)
            # print(labels.shape)
            #Move tensors to the configured device
            inputs = inputs.to(device)
            labels = labels.to(device)

            global_step = epoch * numBatches + i


            #Forward pass
            outputs = model(inputs)
            # print('o', outputs)
            # print('l', labels)
            loss = criterion(outputs, labels)
            # accuracy = calculate_accuracy(outputs,labels)

            #Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            gc.collect()
            

            if(enableTensorboard):
                # img_grid = torchvision.utils.make_grid(images)
                # writer.add_image("images",img_grid, global_step=global_step)
                # writer.add_histogram("fc",model.fc.weight, global_step=global_step)
                writer.add_scalar('training loss', loss.item(), global_step)
                writer.add_scalar('learning rate', scheduler.get_last_lr()[0],global_step)
                # writer.add_scalar('training accuracy', accuracy, global_step)
            
            del inputs, labels, outputs
            print('Batch [{}/{}], Loss {}'.format(i,numBatches,loss.item()))

        scheduler.step()
        print ('Epoch [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, numEpochs, loss.item()))


def testModel(testLoader: DataLoader, model: nn.Module):
   
    model.eval()
    with torch.no_grad():
        numSamples = 0
        error = 0
        for inputs, labels in testLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictedTemps = model(inputs)
            # print("input shape ", inputs.shape)
            # print(predictedTemps.shape)
            # print(labels.shape)
            error = error + torch.abs(predictedTemps - labels).sum().item()
            numSamples = numSamples + labels.size(0)
            del inputs, labels, predictedTemps

        avgError = error/numSamples
        print('Accuracy of the network on the test data is {}'.format(avgError))
    


if __name__ =='__main__':

    dataDir = '/home/artemis/DNN/datasets/GRU/tempData.csv'
    modelPath = "/home/artemis/DNN/GRU/tcnModel.pth"
    writer = SummaryWriter("GRU/runs/tensorboard")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using cuda: ", torch.cuda.is_available())


    batchSize = 64

    trainDataset = CustomTemperatureDataset(dataDir,100,True)
    testDataset = CustomTemperatureDataset(dataDir,100,False)

    print(type(trainDataset[0]))

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=True)

    numEpochs = 200 
    learningRate = 0.01

    model = TemperatureTcn().to(device)
    if os.path.exists(modelPath):
        model.load_state_dict(torch.load(modelPath))

    trainModel(trainLoader,model,numEpochs,learningRate,True)

    torch.save(model.state_dict(),modelPath)

    testModel(testLoader,model)