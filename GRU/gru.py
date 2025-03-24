import numpy as np 
import torch 
import torch.nn as nn 
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os


class CustomTemperatureDataset(Dataset):
    def __init__(self, dataDir, timeSteps, train=False):
        self.dataDir = dataDir
        self.train = train
        self.timeSteps = timeSteps
        self.data = pd.read_csv(dataDir, parse_dates=['Date'], index_col='Date')
        
        self.scaledData, self.dataOffset, self.dataScale = self.scaleMinMax(self.data.values)
        self.inputs, self.lables = self.createDataset(self.scaledData, self.timeSteps)

        if self.train:
            self.inputs = self.inputs[:int(len(self.inputs)*0.8)]
            self.lables = self.lables[:int(len(self.lables)*0.8)]
        else :
            self.inputs = self.inputs[int(len(self.inputs)*0.8):]
            self.lables = self.lables[int(len(self.lables)*0.8):]

        # self.inputs = torch.from_numpy(self.inputs).float()
        # self.lables = torch.from_numpy(self.lables).float()

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.lables[idx]

    @staticmethod
    def scaleMinMax(data: np.array):

        offset = data.min()
        scale = data.max() - offset
        data = (data - offset) / scale
        return data, offset, scale

    @staticmethod
    def createDataset(data, time_step=1):
        print(data)
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            x = data[i:(i+time_step),0]
            X.append(data[i:(i + time_step), 0]) 
            y.append(data[i + time_step, 0]) 

        X = np.array(X)
        y = np.array(y)
        X.reshape(X.shape[0],X.shape[1],1)
        return X,y

class TemperatureGRU(nn.Module):
    def __init__(self):


def trainModel(trainLoader: DataLoader, model : nn.Module, numEpochs, learningRate):
    

    print('')

def testModel(testLoader: DataLoader, model: nn.Module):
    print('')


if __name__ =='__main__':

    dataDir = '/home/artemis/DNN/datasets/GRU/tempData.csv'
    modelPath = "/home/artemis/DNN/models/GRU/temperatureModel.pth"


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using cuda: ", torch.cuda.is_available())


    batchSize = 32

    trainDataset = CustomTemperatureDataset(dataDir,100,True)
    testDataset = CustomTemperatureDataset(dataDir,100,False)

    print(type(trainDataset[0]))

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=True)

    numEpochs = 1 
    learningRate = 0.1

    model = []
    if os.path.exists(modelPath):
        model.load_state_dict(torch.load(modelPath))

    trainModel(trainLoader,model,numEpochs,learningRate);

    torch.save(model.state_dict(),modelPath)

    testModel(testLoader,model)