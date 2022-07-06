from optuna import samplers
from optuna._imports import try_import
import optuna
import statistics
from GA_CL import *
import numpy as np
import globals
#pip install scikit-optimize - Instalar

def objective(trial):
    a = trial.suggest_uniform('a', 2, 10)
    b = trial.suggest_uniform('b', 80, 200)
    c = trial.suggest_uniform('c', 25, 200)
    d = trial.suggest_uniform('d', 35, 200)
    e = trial.suggest_uniform('e', 0.1, 0.9)
    f = trial.suggest_uniform('f', 0.1, 0.9)
     
    Val = GA(globals.n1,globals.n2,globals.maq1,globals.maq2,globals.p1,globals.p2,globals.Prec,a,b,c,d,e,f)

    return Val

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


PATH = 'model.pt'

def save_checkpoint(new_score_track, new_model_state_dict, PATH):
 torch.save({'score_track': new_score_track,
 'net_state_dict': new_model_state_dict
 }, PATH)


def Otimizacao(Instancias):
    ###################################################################################################
    # The objective function is what will be optimized.
    Rest = open("Calibracao.txt","w+")
    myfile = open(Instancias)   #open('Instances.txt')
    next_line = myfile.readline()
    
    while next_line != "" and next_line != "\n":
     
       arquivo = next_line
       arquivo = arquivo.replace("\n", "").replace(" ", "")
    
       print(arquivo)
       
       
       (globals.n1,globals.n2,globals.maq1,globals.maq2,globals.p1,globals.p2,globals.Prec) = Leitura(arquivo)
       
       print(globals.n1,globals.n2,globals.maq1,globals.maq2,globals.p1,globals.p2,globals.Prec)
    
       
       sampler = optuna.integration.SkoptSampler()
       study = optuna.create_study(direction="minimize",sampler=sampler)
         
       trials = max(5,round(globals.n1/2))
       
       print(max(5,round(globals.n1/2)))
       
       study.optimize(objective, n_trials=trials)
       
       
       input_1 = float(globals.n1)
       input_2 = float(statistics.mean(globals.p1))
       input_3 = float(statistics.mean(globals.p2))
       input_4 = float(globals.maq1)
       input_5 = float(globals.maq2)
    
       print(study.best_params)
       
       record = study.best_params
       print(record) # Get statistics
       keys, values = zip(*record.items()) # Split values
       
       
       print(values) # Get statistics
      
       
       # Write data results
       
       print("Gravando...\n")
       print(str(arquivo))
    
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
     
       next_line = myfile.readline()






