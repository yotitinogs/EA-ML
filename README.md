# EA-ML

Data_Set.rar -> Instances provided by "P. M. Cota, B. M. Gimenez, D. P. Araújo, T. H. Nogueira, M. C. de Souza, M. G. Ravetti,
Time-indexed formulation and polynomial time heuristic for a multi-dock truck scheduling problem in a cross-docking centre, Computers & Industrial Engineering 95 (2016) 135–143."

Main.py -> Main file of the EA framework. This procedure uses the following files respecting the indicated order:

1) Kluster.py -> This procedure defines the samples of instances to perform the training of the EA algorithm. They are chosen by KMeans.
2) Optuna.py -> This procedure defines a data set with the optimum EA parameter for each instance's classes.
3) ML.py -> This procedure uses the data set provided by Optuna.py for the training of the ML algorithm.
4) Alg.py -> Initially, the instance is evaluated. If the instance is compatible with the database chosen in step 1 the EA solves the instance problem with its parameters defined by the ML algorithm. Otherwise, the algorithm restarts in step 2, inserting the instance in the most similar cluster. Subsequently, the algorithm is retrained. Finally, the instance is solved by the EA with its parameters defined by the ML algorithm.



