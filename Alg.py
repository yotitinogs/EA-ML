import torch
import os
import torch.nn as nn
import numpy as np
import pandas as pd
import statistics
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from GA import *
from sklearn.preprocessing import StandardScaler    
from Optuna import *
from sklearn import preprocessing
import numpy
from scipy import stats
import globals

def Leitura(nomeArq):

 #Leitura do arquivo
 filename = nomeArq
 dadoE=[]

 with open(filename) as f:
    content = f.read().splitlines()
    for line in content:
        contentBreak = line.split()
        for temp in contentBreak:
            dadoE.append(int(temp))

 n1 = dadoE[0]
 n2 = dadoE[1]
 maq1 = dadoE[2]
 maq2 = dadoE[3]
 p1=[]
 p2=[]

 ind = 0

 for item in range(4,4+n1,1):
    p1.append(dadoE[item])

 for item in range(4+n1,4+n1+n2,1):
    p2.append(dadoE[item])

 Prec = np.zeros(shape=(n2, n1), dtype=int)

 for i in range(0,n2):
   for j in range(4+n1*(i+1)+n2, 4+n1*(i+1)+n2+n1):
     Prec[i,j-(4+n1*(i+1)+n2)]=dadoE[j]

 return n1,n2,maq1,maq2,p1,p2,Prec


def ML_GA(Arquivo1, Arquivo2, tolerance):

    myfile = open(Arquivo1) #open('InstanciaFolder.txt')
    next_line = myfile.readline()
    
    while next_line != "" and next_line != "\n":
     
     myfile2 = open(Arquivo2) #open('Instancia.txt')
     next_line2 = myfile2.readline()
     while next_line2 != "" and next_line2 != "\n":
      
      for indiceF in range(17,18): #17-317
    
       next_line = next_line.rstrip("\n")
       next_line2 = next_line2.rstrip("\n")
       arquivo = "Dados/" + next_line + "/"+ next_line2 + "/" + str(indiceF) + ".dat"
       print(arquivo)
       (globals.n1,globals.n2,globals.maq1,globals.maq2,globals.p1,globals.p2,globals.Prec) = Leitura(arquivo)
    
    
       IND_SIZE = globals.n2
    
       print(globals.n1)
       print(globals.n2)
       print(globals.maq1)
       print(globals.maq2)
       print(globals.p1)
       print(globals.p2)
       print(globals.Prec)
          
          
       input_1 = float(globals.n1)
       input_2 = float(statistics.mean(globals.p1))
       input_3 = float(statistics.mean(globals.p2))
       input_4 = float(globals.maq1)
       input_5 = float(globals.maq2)
       
       print(input_1,input_2,input_3,input_4,input_5)
       
       DataN =[np.array([input_1,input_2,input_3,input_4,input_5]).tolist()]
       
       print(DataN)
       
       D_in = np.genfromtxt("Calibracao.txt", delimiter=',',usecols=range(1,2)).astype(float)
       
       D_in = (int)(D_in[0])
       
       print(D_in)
       
       a =3
       b= 3+D_in 
       
       Base =  np.genfromtxt("Calibracao.txt", delimiter=',',usecols=range(a,b)).astype(float).tolist() 
          
       print(Base)
       
       TestStat = stats.ttest_ind(DataN,Base,1)[1]
          
       print(TestStat)

       print(np.max(TestStat))
       
       if np.max(TestStat) < tolerance:
           
               sampler = optuna.integration.SkoptSampler()
               study = optuna.create_study(direction="minimize",sampler=sampler)
                 
               trials = max(5,round(globals.n1/2))
               
               print(max(5,round(globals.n1/2)))
               
               study.optimize(objective, n_trials=trials)
               
               
#               input_1 = float(globals.n1)
#               input_2 = float(statistics.mean(globals.p1))
#               input_3 = float(statistics.mean(globals.p2))
#               input_4 = float(globals.maq1)
#               input_5 = float(globals.maq2)
            
               print(study.best_params)
               
               record = study.best_params
               print(record) # Get statistics
               keys, values = zip(*record.items()) # Split values
               
               
               print(values) # Get statistics
              
               
               # Write data results
               
               print("Gravando...\n")
               print(str(arquivo))
               
               Rest = open("Calibracao.txt","a+")
            
               Rest.write(str(arquivo))
               Rest.write(",%s"%str(5)) # Parametros Entradas
               Rest.write(",%s"%str(6)) # Parametros Algoritmo
               Rest.write(",%s"%str(round(input_1,2)))
               Rest.write(",%s"%str(round(input_2,2)))
               Rest.write(",%s"%str(round(input_3,2)))
               Rest.write(",%s"%str(round(input_4,2)))
               Rest.write(",%s"%str(round(input_5,2)))
               Rest.write(",%s"%str(round(values[0],0)))
               Rest.write(",%s"%str(round(values[1],0)))
               Rest.write(",%s"%str(round(values[2],0)))
               Rest.write(",%s"%str(round(values[3],0))) 
               Rest.write(",%s"%str(round(values[4],2)))            
               Rest.write(",%s\n"%str(round(values[5],2)))
               
               ML("Calibracao.txt")
               
           
       
       x = torch.from_numpy(np.array([input_1,input_2,input_3,input_4,input_5])).float()
       
       x = np.array(x).reshape ((1,5 )).astype(float)
       x = globals.scalerI.transform(x)
       
       x = torch.from_numpy(x).float()
    
       print(x)
       
       x_out = globals.model(x)
       
       x_out  = x_out.detach().numpy()
       
       print(x_out)
          
       x_out = globals.scalerO.inverse_transform(x_out)
         
       print(x_out)
               
       input_1 = max(round(float(x_out[0][0])),2)
       input_2 = max(round((x_out[0][1])),80)
       input_3 = max(round((x_out[0][2])),25)
       input_4 = max(round((x_out[0][3])),35)
       input_5 = max(float(x_out[0][4]),0.1)  
       input_6 = max(float(x_out[0][5]),0.1)
       
          
       print(input_1,input_2,input_3,input_4,input_5,input_6)
       
    
       ResultGA = GA(globals.n1,globals.n2,globals.maq1,globals.maq2,globals.p1,globals.p2,globals.Prec,input_1,input_2,input_3,input_4,input_5,input_6)          
    
       
       # Write data results
       Rest = open("Resultados.txt","a+")
    
       Rest.write(str(arquivo))
       Rest.write(",PCH_ML")
       Rest.write(",%s"%str(ResultGA))
                             
    
      
      next_line2 = myfile2.readline()
     next_line = myfile.readline()

   
