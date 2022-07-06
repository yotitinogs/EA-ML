from Kluster import *
from Optuna import *
from ML import *
from Alg import *
import globals 

#globals.initialize() 
tolerance = 0.25
arquivoK = "Dados.csv"
arquivoOt = "Instances.txt"
arquivoML = "Calibracao.txt"
arquivoGA1 = "InstanciaFolder.txt"
arquivoGA2 = "Instancia.txt"
Clusterizacao(arquivoK)
Otimizacao(arquivoOt)
ML(arquivoML)
ML_GA(arquivoGA1, arquivoGA2, tolerance) 