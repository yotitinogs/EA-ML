import torch
import os
import torch.nn as nn
import numpy as np
import pandas as pd
import statistics
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
from sklearn import preprocessing
import globals
PATH = 'model.pt'

def save_checkpoint(new_score_track, new_model_state_dict, PATH):
 torch.save({'score_track': new_score_track,
 'net_state_dict': new_model_state_dict
 }, PATH)


def MachineLearning(nomeArq, n, DIn, DOut, PATH):

 N, D_in, H, D_out = n, DIn, 1000, DOut

 training = np.genfromtxt("Calibracao.txt", delimiter=',').astype(float) 

 print(training)
 
 print(D_in)
 
 a =3
 b= 3+D_in   
 training_inputs = np.genfromtxt("Calibracao.txt", delimiter=',',usecols=range(a,b)).astype(float) 
      
 print(training_inputs)

 a =3+D_in
 b= 3+D_in+D_out
 training_outputs = np.genfromtxt("Calibracao.txt", delimiter=',',usecols=range(a,b)).astype(float) 

 print(training_outputs)
 
 print(D_out)

 training_outputs = training_outputs.reshape((-D_out, D_out)).astype(float) 
     
 print(training_outputs)
  
 # Create random Tensors to hold inputs and outputs
 x = torch.from_numpy(training_inputs).float()

 y = torch.from_numpy(training_outputs).float()

 print(x)
 print(y)
 
 min_max_scalerI = preprocessing.MinMaxScaler()
 x = min_max_scalerI.fit_transform(x)
 min_max_scalerO = preprocessing.MinMaxScaler()
 y = min_max_scalerO.fit_transform(y)


 print(x)
 print(y)
 
 
 x = torch.from_numpy(x).float()
 y = torch.from_numpy(y).float()

 # Use the nn package to define our model and loss function.
 #model = torch.nn.Sequential(
 #   torch.nn.Linear(D_in, H),
 #   torch.nn.ReLU(),
 #   torch.nn.Linear(H, D_out),
 #)
 
 
 
 #create a network with dropout
 model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.Dropout(0.5), #50 % probability 
    nn.ReLU(),
    nn.Linear(H, H),
    nn.Dropout(0.2), #20% probability
    nn.ReLU(),
    nn.Linear(H, D_out),
 )
 
 learning_rate = 1e-4
 optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

 save_checkpoint(0,model.state_dict(), PATH)
 checkpoint = torch.load(PATH)
 model.load_state_dict(checkpoint['net_state_dict'])
 return model,min_max_scalerI , min_max_scalerO
 
#Start loop
 
def ML(Arquivo):
 
    filename = Arquivo
    #'Calibracao.txt'
    dadoE=[]
    
    with open(filename) as f:
        content = f.read().splitlines()
        for line in content:
            contentBreak = line.split(",")
            for temp in contentBreak:
                dadoE.append(temp)
            break    
                
    print(dadoE)            
    
    DIn = (int)(dadoE[1])
    
    DOut = (int)(dadoE[2])
    
    fileObject = open(filename)
    n = sum(1 for row in fileObject) 
    
    print(n,DIn,DOut)
    
    globals.model,globals.scalerI,globals.scalerO = MachineLearning('Calibracao.txt', n, DIn, DOut, PATH)
    
    
    #return  model,scalerI,scalerO

   
    
    #Test - The oustput is the model.pt
    
#    input_1 = float(30)
#    input_2 = float(61)
#    input_3 = float(58)
#    input_4 = float(4)
#    input_5 = float(4)
#    
#    x = torch.from_numpy(np.array([input_1,input_2,input_3,input_4,input_5])).float()
#       
#    x = np.array(x).reshape ((1,5 )).astype(float)
#    
#    x = globals.scalerI.transform(x)
#       
#    x = torch.from_numpy(x).float()
#    
#    x_out = globals.model(x)
#       
#    x_out  = x_out.detach().numpy()
#       
#    print(x_out)
#          
#    x_out = globals.scalerO.inverse_transform(x_out)
#         
#    print(x_out)