import array
import random
import numba
import os

import numpy as np
import multiprocessing

from deap import algorithms
from deap import cma
from deap import base
from deap import creator
from deap import tools
from scoop import futures
from timeit import default_timer as timer
import sys
import matplotlib.pyplot as plt


def takeSecond(elem):
 return elem[1]


def evalPop(individual):   
    r= []
    cont =0
    for i in range(0, n2):
      for j in range(0, n1):
       if Prec[individual[i],j]==1:
        r.append(cont)
        r[cont] = (i,individual[i],j,-p1[j])
        cont+=1

    r_initial=[]
    r.sort(key=lambda t: (t[0], t[3]))

    for a, b, c, d in r:
     r_initial.append(c)

    r_initial = list(dict.fromkeys(r_initial))
 
    rc = np.zeros(shape=(n2), dtype=int)
    C = np.zeros(shape=(n1), dtype=int)
    Maq = np.zeros(shape=(maq1, n1), dtype=int)
    MaqAc =  np.zeros(shape=(maq1), dtype=int)
    MaqPos = np.zeros(shape=(maq1), dtype=int) 


    for i in range(0,n1):
     k = np.argmin(MaqAc,axis=0)
     MaqAc[k] = MaqAc[k] + p1[r_initial[i]]
     Maq[k][MaqPos[k]] = r_initial[i]
     C[r_initial[i]]= MaqAc[k]
     MaqPos[k] +=1 

    for a, b, c, d in r:
     rc[b] = max(rc[b],C[c])

    C2 = np.zeros(shape=(n2), dtype=int)
    Maq2 = np.zeros(shape=(maq2, n2), dtype=int)
    MaqAc2 =  np.zeros(shape=(maq2), dtype=int)
    MaqPos2 = np.zeros(shape=(maq2), dtype=int)
    p2aux = p2[:]
    rcaux = rc[:]

    for i in range(0,n2):
     k = np.argmin(MaqAc2,axis=0)
     p2MAX = -1
     ind = -1
     for j in range(0,n2):
        if rc[j] <= MaqAc2[k] and p2aux[j] > p2MAX :
           p2MAX = p2aux[j]
           ind = j
     if ind == -1:
        p2MAX = -1
        ind = np.argmin(rcaux,axis=0)
        rMIN = rcaux[ind]
        for j in range(0,n2):
           if rcaux[j] == rMIN and p2aux[j] > p2MAX :
            p2MAX = p2aux[j]
            rMIN = rcaux[j]
            ind = j
     MaqAc2[k] = max(MaqAc2[k],rc[ind]) + p2[ind]
     Maq2[k][MaqPos2[k]] = ind
     C2[ind]= MaqAc2[k]
     MaqPos2[k] +=1
     p2aux[ind] = -2
     rcaux[ind] = 1000000

    Cmax = max(C2)
    return Cmax,

def Solucao_Inicial():

 pc = []

 for i in range(0,n1):
  pc.append(0)
  maxx = -1
  for j in range(0,n2):
   if Prec[j][i] == 1 :
     pc[i] = pc[i] + p2[j]
     if p2[j] > maxx:
      maxx = p2[j]
  if (float)(pc[i]/maq2) < maxx :
   pc[i] = maxx
  else:
   pc[i] = (float)(pc[i]/maq2)

 r=[]

 for i in range(0, n1):
  r.append(i)
  r[i] = (i, pc[i], p1[i])

 r.sort(reverse=True, key=lambda t: (t[1], t[2]))


 PCH = 0
 Pacum = 0
 sol1 = []
 sol0 = []

 for key, value1, value2 in r:
  Pacum = Pacum + value2
  PCH = max(Pacum + value1,PCH)
  sol1.append(key)

 sol0[:] = sol1

 #Busca Local

 FOMIN = 100000000
 sol2B = []

 i1 = 0
 while i1 < n1-1:
		i2 = i1
		while i2 <= n1-1:
    
				aux = sol1[i1]
				sol1[i1]=sol1[i2]
				sol1[i2]= aux
    
				PCH0 = p1[sol1[0]] + pc[sol1[0]]
				PCHC = PCH0 + (PCH - PCH0)/maq1
    

				C = np.zeros(shape=(n1), dtype=float)
				Maq = np.zeros(shape=(maq1, n1), dtype=int)
				MaqAc =  np.zeros(shape=(maq1), dtype=float)
				MaqPos = np.zeros(shape=(maq1, n1), dtype=int) 

				k = 0

				for i in range(0,n1):
					if MaqAc[k] + p1[sol1[i]] + pc[sol1[i]]  >= PCHC and k < maq1-1:
						 k = k+1
					MaqAc[k] = MaqAc[k] + p1[sol1[i]]
					Maq[k][MaqPos[k]] = sol1[i]
					C[sol1[i]]= MaqAc[k]
					MaqPos[k] +=1 

				rc = np.zeros(shape=(n2), dtype=float)

				for i in range(0,n2):
						 for j in range(0,n1):
						   if Prec[i][j] == 1: 
						    rc[i] = max(rc[i],C[j])


				C2 = np.zeros(shape=(n2), dtype=int)
				Maq2 = np.zeros(shape=(maq2, n2), dtype=int)
				MaqAc2 =  np.zeros(shape=(maq2), dtype=int)
				MaqPos2 = np.zeros(shape=(maq2, n2), dtype=int)
				p2aux = p2.copy()
				rcaux = rc.copy()

				for i in range(0,n2):
						k = np.argmin(MaqAc2,axis=0)
						p2MAX = -1
						ind = -1
						for j in range(0,n2):
						   if rc[j] <= MaqAc2[k] and p2aux[j] > p2MAX :
						      p2MAX = p2aux[j]
						      ind = j
						if ind == -1:
						   p2MAX = -1
						   ind = np.argmin(rcaux,axis=0)
						   rMIN = rcaux[ind]
						   for j in range(0,n2):
						      if rcaux[j] == rMIN and p2aux[j] > p2MAX :
						       p2MAX = p2aux[j]
						       rMIN = rcaux[j]
						       ind = j
						MaqAc2[k] = max(MaqAc2[k],rc[ind]) + p2[ind]
						Maq2[k][MaqPos2[k]] = ind
						C2[ind]= MaqAc2[k]
						MaqPos2[k] +=1
						p2aux[ind] = -2
						rcaux[ind] = 1000000

				r2=[]
				rc2 = np.zeros(shape=(n2), dtype=int)

				for i in range(0, n2):
					r2.append(i)
					rc2[i] = C2[i]-p2[i]
					r2[i] = (i, rc2[i], -p2[i])

				r2.sort(key=lambda t: (t[1], t[2]))

				sol2 = []

				for key, value1, value2 in r2:
					sol2.append(key)

				CMAXBL = max(C2)
				
				if CMAXBL < FOMIN:
					FOMIN = CMAXBL
					sol2B[:] = sol2
					i1 = 0
					i2 = 0
				else:
					sol1[i2]= sol1[i1]
					sol1[i1] = aux
	
				i2 = i2 +1

		i1 = i1 + 1 

 print (FOMIN)
 return  sol2B


def customize_individual():
 return Initial

def load_individuals(creator,n):
    individuals = []
    pc = []

    for i in range(0,n1):
					pc.append(0)
					maxx = -1
					for j in range(0,n2):
						if Prec[j][i] == 1 :
						  pc[i] = pc[i] + p2[j]
						  if p2[j] > maxx:
						   maxx = p2[j]
					if (float)(pc[i]/maq2) < maxx :
						pc[i] = maxx
					else:
						pc[i] = (float)(pc[i]/maq2)

    r=[]

    for i in range(0, n1):
					r.append(i)
					r[i] = (i, pc[i], p1[i])

    r.sort(reverse=True, key=lambda t: (t[1], t[2]))


    PCH = 0
    Pacum = 0
    sol1 = []
    sol0 = []

    for key, value1, value2 in r:
					Pacum = Pacum + value2
					PCH = max(Pacum + value1,PCH)
					sol1.append(key)

    sol0[:] = sol1

				#Busca Local

    FOMIN = 100000000
    sol2B = []

    i1 = 0
    while i1 < n1-1:
					i2 = i1
					while i2 <= n1-1:
						 
							aux = sol1[i1]
							sol1[i1]=sol1[i2]
							sol1[i2]= aux
						 
							PCH0 = p1[sol1[0]] + pc[sol1[0]]
							PCHC = PCH0 + (PCH - PCH0)/maq1
						 

							C = np.zeros(shape=(n1), dtype=float)
							Maq = np.zeros(shape=(maq1, n1), dtype=int)
							MaqAc =  np.zeros(shape=(maq1), dtype=float)
							MaqPos = np.zeros(shape=(maq1, n1), dtype=int) 

							k = 0

							for i in range(0,n1):
								if MaqAc[k] + p1[sol1[i]] + pc[sol1[i]]  >= PCHC and k < maq1-1:
										k = k+1
								MaqAc[k] = MaqAc[k] + p1[sol1[i]]
								Maq[k][MaqPos[k]] = sol1[i]
								C[sol1[i]]= MaqAc[k]
								MaqPos[k] +=1 

							rc = np.zeros(shape=(n2), dtype=float)

							for i in range(0,n2):
										for j in range(0,n1):
												if Prec[i][j] == 1: 
												 rc[i] = max(rc[i],C[j])


							C2 = np.zeros(shape=(n2), dtype=int)
							Maq2 = np.zeros(shape=(maq2, n2), dtype=int)
							MaqAc2 =  np.zeros(shape=(maq2), dtype=int)
							MaqPos2 = np.zeros(shape=(maq2, n2), dtype=int)
							p2aux = p2.copy()
							rcaux = rc.copy()

							for i in range(0,n2):
									k = np.argmin(MaqAc2,axis=0)
									p2MAX = -1
									ind = -1
									for j in range(0,n2):
												if rc[j] <= MaqAc2[k] and p2aux[j] > p2MAX :
												   p2MAX = p2aux[j]
												   ind = j
									if ind == -1:
												p2MAX = -1
												ind = np.argmin(rcaux,axis=0)
												rMIN = rcaux[ind]
												for j in range(0,n2):
												   if rcaux[j] == rMIN and p2aux[j] > p2MAX :
												    p2MAX = p2aux[j]
												    rMIN = rcaux[j]
												    ind = j
									MaqAc2[k] = max(MaqAc2[k],rc[ind]) + p2[ind]
									Maq2[k][MaqPos2[k]] = ind
									C2[ind]= MaqAc2[k]
									MaqPos2[k] +=1
									p2aux[ind] = -2
									rcaux[ind] = 1000000

							r2=[]
							rc2 = np.zeros(shape=(n2), dtype=int)

							for i in range(0, n2):
								r2.append(i)
								rc2[i] = C2[i]-p2[i]
								r2[i] = (i, rc2[i], -p2[i])

							r2.sort(key=lambda t: (t[1], t[2]))

							sol2 = []

							for key, value1, value2 in r2:
								sol2.append(key)

							CMAXBL = max(C2)
							
							if CMAXBL < FOMIN:
								FOMIN = CMAXBL
								sol2B[:] = sol2
								i1 = 0
								i2 = 0
								if Improved == 1:
									individual = creator(sol2)
									individuals.append(individual)
							else:
								sol1[i2]= sol1[i1]
								sol1[i1] = aux

       # Sem re-start
							#sol1[i2]= sol1[i1]
							#sol1[i1] = aux

							if Improved == 0:
								individual = creator(sol2)
								individuals.append(individual)

							i2 = i2 +1

					i1 = i1 + 1 
    
    print(FOMIN)
    return individuals



def customize_population_config(creator,toolbox):
	print("Select customize population")
	toolbox.register("population", load_individuals,  creator.Individual)

def customize_individual_config(creator,toolbox):
	# obviously we generate more elaborate individuals
	print("Select customize individual")
	global Initial 
	Initial = Solucao_Inicial()
	toolbox.register("individual", tools.initIterate, creator.Individual, customize_individual)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def random_population_config(creator,toolbox):
	print("Select random population")
	# Attribute generator
	toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)
	# Individual generator
	toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)


  
def GA(n1g,n2g,maq1g,maq2g,p1g,p2g,Precg,input_1,input_2,input_3,input_4,input_5,input_6):


   global Improved
   Improved = 0
   global n1
   n1 = n1g
   global n2
   n2 = n2g
   global maq1
   maq1 = maq1g
   global maq2
   maq2 = maq2g
   global p1
   p1 = p1g
   global p2
   p2 = p2g
   global Prec
   Prec = Precg
   
   global IND_SIZE
   IND_SIZE = n2g
      
   creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
   creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

   toolbox = base.Toolbox()
   toolbox.register("map", futures.map)

  #we execute the function depends on the configuration we need
   customize_population_config(creator,toolbox)

   #toolbox.register("mate", tools.cxPartialyMatched)
   toolbox.register("mate", tools.cxOrdered)
   toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/IND_SIZE)
   toolbox.register("evaluate", evalPop)

   print ("Running in " + str(multiprocessing.cpu_count()) + " processing")
   pool = multiprocessing.Pool()
   toolbox.register("map", pool.map)
   #python3 -m scoop nome.py


   tamTour = int(input_1) #3
   NGEN = int(input_2) #150
   MU = int(input_3) #75
   LAMBDA = int(input_4) #100

   if float(input_5) + float(input_6) > 1:
     ponderado = float(input_5) + float(input_6)
     input_5 = float(input_5/ponderado)
     input_6 = float(input_6/ponderado)

   CXPB = round(input_5,3) #0.95
   MUTPB = round(input_6,3) #0.4
   
   print(CXPB,MUTPB)

   ####################################################################

   start = timer()
   random.seed(169)

   verbose=__debug__

   stats = tools.Statistics(lambda ind: ind.fitness.values)
   stats.register("avg", np.mean)
   stats.register("std", np.std)
   stats.register("min", np.min)
   stats.register("max", np.max)

   logbook = tools.Logbook()
   logbook.header = ['gen', 'nevals', 'globalBest'] + (stats.fields if stats else [])

   pop = toolbox.population(n1*n2)
   halloffame = tools.HallOfFame(1)

   # Evaluate the individuals with an invalid fitness
   invalid_ind = [ind for ind in pop if not ind.fitness.valid]

   # Evaluate the entire population
   fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
   for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = fit

   if halloffame is not None:
      halloffame.update(pop)
      
   ResultGA = [0,0,0]
   LimitNI = NGEN*0.25
   NoImprovement = 1
   GlobalBest = tools.selBest(pop, 1)[0].fitness.values[0]
   CurrentFit = GlobalBest
   ResultGA[0] = GlobalBest
   ResultGA[1] = 0
   
   
   record = stats.compile(pop) if stats else {}
   logbook.record(gen=0, nevals=len(invalid_ind), globalBest=GlobalBest, **record)

   # Begin the generational process
   iterPlt=1
   MTD = 0

   for gen in range(1,NGEN + 1):

        # Vary the population
        offspring = algorithms.varOr(pop, toolbox, LAMBDA, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid] 
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
          ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
          halloffame.update(offspring)

        # Select the next generation population
        toolbox.register("select", tools.selTournament, tournsize=tamTour)
        pop[:] = toolbox.select(pop + offspring, MU)

        # Update global best
        if tools.selBest(pop, 1)[0].fitness.values[0] < GlobalBest:
          GlobalBest = tools.selBest(pop, 1)[0].fitness.values[0]
          ResultGA[0] = GlobalBest
          itBest = gen
          ResultGA[1] = itBest
          NoImprovement = 1
          toolbox.register("mate", tools.cxOrdered)
          MTD = 0
        else:
          NoImprovement = NoImprovement + 1

        # Stop criteria
        if NoImprovement >= LimitNI:
          NoImprovement = 1
          if MTD == 0:
            toolbox.register("mate", tools.cxPartialyMatched)
            MTD = 1
          else:
            toolbox.register("mate", tools.cxOrdered)
            MTD = 0
        #if NoImprovement >= LimitNI:
          #break
        
        # Append the current generation statistics to the logbook
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), globalBest = GlobalBest, **record)

   end = timer()
   
   elapsed_time = end-start
   ResultGA[2] = elapsed_time
   print("Elapsed time:")
   print(elapsed_time)

   record = stats.compile(pop)
   print(record) # Get statistics
   keys, values = zip(*record.items()) # Split values

   pool.close()
   
   print(GlobalBest)
   
   return ResultGA[0]




