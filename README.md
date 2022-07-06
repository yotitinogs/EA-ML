# EA-ML
Main.py -> Main file of the EA framework. This procedure uses the following files respecting the indicated order.

1) Kluster.py -> This procedure defines the samples of instances to perform the training of the EA algorithm. They are chosen by KMeans. 
2) Optuna.py -> This procedure defines an data set with the optimum EA parameter for each instances classes.
3) ML.py ->   This procedure uses the data set provided by Optuna.py for the training of the ML algorithm.
4) Alg.py -> Inittially the instance is evaluated. If the instance is compatible with the database chosen in step 1 the EA solves 
the instance problem with its parameters defined by the ML algorithm. Otherwise, the algorithm restarts in step 2, inserting the
instance in the most similar cluster. Subsequently, the algorithm is retrained. Finally,  the instance is solved by the EA with 
its parameters defined by the ML algorithm.


