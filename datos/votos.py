# Aplicaciones de la Inteligencia Artificial
# Máster en Ingeniería Informática 
# Universidad de Sevilla 
# votos.py

# Datos sobre votos de cada uno de los 435 congresitas de Estados
# Unidos en 17 votaciones realizadas durante 1984, clasificados según su
# partido (republicano o demócrata). 

# Tomado de 
# http://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records

# El valor de cada votación lo codificamos numéricamente 
# de la siguiente manera: 

# 1: voto sí
# -1: voto no
# 0: "Present" (similar a la abstención)

import numpy as np

votos_clases=['republicano','democrata']


votos_entr= np.array([[-1,1,-1,1,1,1,-1,-1,-1,1,0,1,1,1,-1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,0],
            [0,1,1,0,1,1,-1,-1,-1,-1,1,-1,1,1,-1,-1],
            [-1,1,1,-1,0,1,-1,-1,-1,-1,1,-1,1,-1,-1,1],
            [1,1,1,-1,1,1,-1,-1,-1,-1,1,0,1,1,1,1],
            [-1,1,1,-1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,-1,0,1,1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,0,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,0,0],
            [-1,1,-1,1,1,-1,-1,-1,-1,-1,0,0,1,1,-1,-1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,1,0,1,1,0,0],
            [-1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,0,0],
            [1,1,1,-1,-1,1,1,1,0,1,1,0,-1,-1,1,0],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,0,0,-1,0],
            [-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,0,-1,0],
            [1,-1,1,-1,-1,1,-1,1,0,1,1,1,0,-1,-1,1],
            [1,0,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,0,1,1,-1,-1],
            [1,1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1],
            [1,1,1,-1,-1,0,1,1,-1,-1,1,-1,-1,-1,1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,0,0,1,1],
            [1,0,1,-1,-1,-1,1,1,1,-1,-1,0,-1,-1,1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1],
            [1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,0],
            [1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1],
            [1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1],
            [1,-1,-1,1,1,-1,1,1,1,-1,-1,1,1,1,-1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
            [1,1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,0],
            [1,1,1,-1,-1,-1,1,1,1,1,-1,-1,1,-1,1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
            [1,0,-1,1,1,1,-1,-1,-1,1,-1,1,0,1,-1,1],
            [1,1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,-1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1],
            [1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,0,-1,-1,-1,-1,0],
            [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,0],
            [1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1],
            [1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,0],
            [1,1,1,-1,-1,-1,1,1,0,-1,1,-1,-1,-1,1,0],
            [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1],
            [1,-1,1,-1,-1,-1,1,1,0,-1,-1,-1,-1,-1,-1,0],
            [1,1,1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1],
            [-1,0,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
            [1,1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1],
            [-1,1,-1,1,1,1,-1,0,-1,-1,-1,1,1,1,-1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,0,0],
            [1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1],
            [1,1,1,-1,-1,1,0,1,-1,-1,1,1,-1,1,-1,0],
            [-1,1,-1,1,1,1,-1,-1,-1,1,1,1,1,1,-1,-1],
            [-1,1,-1,1,1,1,-1,-1,-1,1,1,1,1,1,-1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,0],
            [1,1,1,-1,-1,0,1,1,1,1,-1,-1,-1,-1,1,0],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
            [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,0],
            [1,1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,1,0,-1,-1,-1,1],
            [1,1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,-1,1],
            [-1,1,-1,1,1,1,1,-1,-1,-1,1,1,1,1,-1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1],
            [1,0,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1],
            [1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,0],
            [1,1,1,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,0],
            [1,-1,1,1,1,-1,1,-1,1,1,-1,-1,1,1,-1,1],
            [1,-1,1,-1,-1,1,1,1,1,1,1,-1,-1,1,1,1],
            [-1,1,1,1,1,1,-1,-1,-1,1,1,-1,1,1,-1,-1],
            [-1,1,1,-1,1,1,-1,-1,-1,1,1,1,1,1,-1,0],
            [-1,1,1,1,1,1,-1,1,1,1,1,1,1,1,-1,1],
            [1,1,1,-1,1,1,-1,-1,-1,1,1,-1,1,1,-1,1],
            [-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,1,-1,-1],
            [1,-1,1,-1,-1,1,1,1,1,1,-1,1,-1,1,-1,0],
            [1,-1,1,-1,-1,-1,1,1,0,1,1,1,-1,1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
            [-1,0,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1],
            [-1,-1,1,-1,1,1,-1,-1,-1,1,1,1,1,1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
            [-1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,0,1,1,-1,0],
            [1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1],
            [1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1],
            [1,1,1,-1,-1,-1,1,1,-1,1,1,-1,-1,0,1,1],
            [1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,1],
            [1,-1,1,-1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1],
            [1,-1,1,-1,1,1,-1,0,0,-1,1,0,0,0,1,1],
            [-1,-1,0,-1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1],
            [1,-1,-1,-1,1,1,1,-1,-1,1,1,-1,-1,1,-1,1],
            [1,1,1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,1,0,1,1,1,-1,-1],
            [1,-1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,-1,1],
            [1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,1,-1,1],
            [1,-1,1,-1,1,1,1,-1,0,-1,1,-1,1,1,1,0],
            [1,-1,-1,-1,1,1,0,-1,0,-1,-1,-1,-1,1,0,-1],
            [0,0,0,0,-1,1,1,1,1,1,0,-1,1,1,-1,0],
            [1,1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1,-1,1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1],
            [-1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
            [1,0,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,0],
            [1,0,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,0],
            [-1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1],
            [-1,0,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
            [-1,0,1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,0],
            [-1,0,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1],
            [1,0,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,0],
            [-1,0,1,-1,0,0,1,1,1,1,0,0,-1,-1,1,1],
            [1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1],
            [1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,-1,1],
            [-1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
            [-1,0,0,1,1,1,-1,-1,-1,1,-1,1,1,1,0,1],
            [-1,0,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,1],
            [1,0,-1,1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1],
            [-1,0,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1],
            [-1,0,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
            [-1,0,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
            [-1,0,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,1,1],
            [-1,0,1,-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,1],
            [0,0,1,-1,-1,-1,1,1,0,-1,0,0,0,0,0,0],
            [1,0,1,-1,0,0,1,1,1,-1,-1,-1,-1,-1,1,0],
            [-1,-1,1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,0],
            [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,0],
            [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1],
            [-1,1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,1],
            [-1,0,1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1,1],
            [-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,-1,1,-1,1],
            [1,-1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,1],
            [-1,-1,-1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,1],
            [-1,-1,-1,1,1,1,1,1,1,1,-1,1,1,1,0,1],
            [-1,-1,-1,1,1,1,1,1,1,1,-1,1,1,1,-1,1],
            [0,1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,1,1,1],
            [-1,0,-1,-1,-1,1,1,1,1,1,-1,-1,-1,1,-1,0],
            [-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,-1,1,0,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1],
            [-1,-1,-1,-1,-1,-1,1,1,1,1,-1,1,1,1,1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,1,1,1,1,1,-1,1],
            [-1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,1,-1,1,1],
            [1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
            [1,1,0,1,1,1,-1,-1,1,-1,1,0,1,1,-1,-1],
            [-1,1,1,-1,-1,1,-1,1,1,1,1,-1,1,-1,1,1],
            [-1,-1,1,-1,-1,1,1,1,1,1,1,-1,1,1,-1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
            [1,1,-1,1,1,1,-1,0,-1,-1,1,1,1,1,-1,-1],
            [1,1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,-1,-1],
            [-1,1,1,-1,-1,1,-1,1,1,-1,1,-1,0,0,0,0],
            [-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1],
            [-1,1,1,-1,0,1,1,1,1,1,1,-1,-1,0,-1,0],
            [-1,1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1],
            [-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,1,1,-1,1],
            [-1,1,1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,1],
            [-1,1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,-1,1],
            [1,1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,-1,0],
            [-1,1,1,-1,-1,1,1,1,1,1,1,-1,1,-1,1,0],
            [1,-1,1,1,1,1,1,1,-1,1,-1,1,-1,1,1,1],
            [1,-1,1,1,1,1,1,1,-1,1,1,1,-1,1,1,1],
            [-1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,1,1,1,0],
            [1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,1],
            [1,-1,1,-1,-1,-1,0,1,1,0,-1,-1,-1,-1,1,0],
            [-1,0,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
            [-1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1],
            [-1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,-1,1],
            [1,0,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,0],
            [-1,1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1],
            [-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,-1,1,1,1],
            [-1,-1,1,-1,-1,-1,1,1,1,1,1,0,-1,-1,1,1],
            [0,-1,1,-1,-1,-1,1,1,1,1,1,0,-1,-1,1,0],
            [1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1],
            [0,0,1,-1,-1,-1,1,1,1,0,0,-1,-1,-1,0,0],
            [-1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1],
            [1,0,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1],
            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
            [-1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1],
            [1,-1,1,-1,-1,-1,1,1,1,1,-1,0,-1,-1,1,1],
            [-1,1,1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1],
            [1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,0],
            [1,0,-1,1,1,1,1,1,-1,-1,-1,1,0,1,0,0],
            [1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1],
            [-1,0,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,0],
            [-1,1,-1,1,1,1,-1,0,-1,1,-1,1,1,1,-1,0],
            [-1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,1,1,1],
            [-1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1],
            [-1,-1,1,-1,-1,1,1,0,1,1,1,-1,-1,-1,1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,0],
            [-1,-1,1,-1,-1,1,1,1,1,-1,1,1,-1,1,1,0],
            [-1,0,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1],
            [-1,-1,1,-1,-1,-1,1,1,1,1,1,-1,0,-1,1,0],
            [1,1,-1,-1,-1,-1,1,1,0,-1,1,-1,-1,-1,1,0],
            [-1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1],
            [1,-1,1,-1,-1,1,1,1,1,1,1,-1,-1,-1,1,1],
            [1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1],
            [-1,-1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,-1,1],
            [-1,-1,1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,1],
            [1,0,-1,1,1,1,1,-1,-1,1,-1,1,1,1,-1,1],
            [-1,-1,1,-1,-1,-1,1,1,1,-1,-1,0,-1,-1,1,1],
            [1,1,1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,1],
            [-1,-1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
            [-1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1],
            [-1,1,1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,1,1],
            [1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
            [-1,1,1,1,1,1,-1,-1,-1,1,1,1,1,1,1,0],
            [1,1,1,-1,1,1,-1,-1,0,1,-1,-1,-1,1,1,0],
            [-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1],
            [1,0,1,-1,-1,-1,1,1,1,-1,0,-1,-1,-1,1,0],
            [-1,1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1,1,1,1],
            [-1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,0],
            [-1,1,1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,0],
            [1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,0],
            [-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,1,-1,0,1,1,-1,-1],
            [-1,0,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1],
            [-1,-1,1,-1,-1,1,1,1,1,-1,1,-1,-1,1,1,1],
            [1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,0,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,0,-1,1],
            [-1,1,1,1,1,1,1,-1,1,1,-1,1,1,1,-1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1],
            [-1,1,-1,1,1,1,-1,-1,1,1,-1,1,1,1,-1,1],
            [-1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,-1,1,0],
            [-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
            [-1,-1,1,-1,-1,1,1,1,1,1,-1,1,-1,1,1,0],
            [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,1],
            [-1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1],
            [1,-1,1,-1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,1],
            [1,1,1,-1,-1,-1,1,1,0,1,-1,-1,-1,-1,1,0],
            [-1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,1,1,1],
            [-1,-1,-1,1,-1,1,1,0,1,-1,-1,1,1,1,-1,1],
            [1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,1,1],
            [-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,0,-1,1,1,1],
            [-1,1,1,-1,-1,-1,1,1,0,1,-1,-1,1,-1,1,1],
            [1,-1,1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,1],
            [1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1],
            [-1,-1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,0,1],
            [-1,1,-1,1,1,1,0,-1,-1,-1,-1,0,1,1,-1,-1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,-1,1,-1,-1,-1,1,1,0,-1,1,-1,-1,-1,1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
            [1,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1],
            [1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1],
            [1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,1,1],
            [-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1],
            [-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,1,0,1],
            [-1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1],
            [1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1],
            [1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1],
            [1,-1,1,-1,-1,0,1,1,1,-1,0,0,-1,0,0,0],
            [1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1],
            [1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,0],
            [1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,0],
            [1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,1],
            [1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,-1,1],
            [1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,0],
            [1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1],
            [-1,1,1,-1,-1,1,1,1,1,-1,0,-1,-1,-1,-1,1],
            [1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,0],
            [-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,1,0,1],
            [1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,1],
            [-1,-1,1,-1,1,1,-1,-1,-1,-1,0,-1,1,1,-1,-1],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,-1],
            [-1,-1,1,1,1,1,1,1,-1,1,-1,-1,-1,1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1]])

votos_entr_clas=['republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'republicano',
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'republicano',
                 'republicano',
                 'republicano',
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'republicano',
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'republicano',
                 'republicano',
                 'republicano',
                 'republicano',
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'republicano',
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'republicano',
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'republicano',
                 'republicano',
                 'republicano']



votos_valid=[[-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1],
             [-1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,-1,1],
             [1,-1,1,1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,0],
             [1,-1,-1,1,1,1,-1,-1,-1,1,-1,0,1,1,-1,-1],
             [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1],
             [-1,-1,1,-1,-1,1,1,1,1,1,1,-1,-1,-1,0,1],
             [-1,-1,1,-1,-1,1,1,1,1,1,1,-1,-1,-1,1,1],
             [-1,-1,1,-1,-1,1,0,1,0,1,1,1,-1,1,1,0],
             [1,1,1,0,-1,1,1,1,1,-1,1,-1,1,-1,0,1],
             [1,1,1,-1,1,1,-1,1,-1,1,1,-1,1,1,1,1],
             [1,1,1,-1,1,1,-1,1,-1,1,1,-1,1,1,-1,0],
             [1,-1,1,-1,0,1,0,1,1,1,-1,-1,1,1,-1,1],
             [1,-1,1,-1,-1,1,1,1,1,1,-1,0,-1,1,-1,1],
             [1,-1,1,-1,-1,1,1,1,-1,1,1,-1,1,1,1,1],
             [1,1,1,-1,-1,1,1,1,1,1,1,-1,1,1,1,1],
             [-1,1,1,-1,-1,1,1,1,-1,1,1,-1,1,1,-1,0],
             [-1,1,-1,1,1,1,0,0,-1,1,-1,1,0,0,0,0],
             [-1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,1,1],
             [1,1,1,-1,-1,1,1,1,1,1,-1,-1,0,-1,1,0],
             [-1,1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1],
             [-1,1,1,-1,-1,1,1,1,1,1,-1,-1,1,1,1,1],
             [-1,-1,-1,1,1,-1,1,1,1,1,-1,1,1,1,-1,1],
             [-1,-1,0,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,1],
             [-1,-1,-1,1,1,1,1,-1,-1,1,-1,1,1,1,-1,1],
             [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
             [-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,0],
             [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1],
             [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
             [1,-1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,-1,0],
             [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1],
             [1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1,1],
             [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,1,-1],
             [-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,1],
             [1,1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1],
             [-1,1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
             [-1,1,-1,1,1,1,1,1,-1,-1,1,1,1,1,1,1],
             [-1,1,1,1,1,1,1,0,-1,-1,-1,-1,0,0,1,0],
             [-1,-1,-1,-1,-1,1,-1,1,1,-1,1,1,1,1,1,-1],
             [1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1],
             [-1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,0],
             [1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,0],
             [-1,1,1,-1,-1,1,-1,1,1,1,-1,-1,1,1,-1,1],
             [1,1,1,-1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,1],
             [1,1,1,-1,0,1,-1,0,-1,-1,1,-1,1,1,-1,0],
             [1,1,1,-1,1,1,-1,1,0,1,-1,-1,1,1,-1,0],
             [-1,1,-1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1],
             [-1,1,-1,-1,1,1,-1,-1,0,-1,-1,1,1,1,-1,1],
             [1,1,-1,1,-1,-1,1,1,1,-1,1,-1,-1,1,-1,1],
             [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1],
             [1,1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1],
             [1,0,1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,1,0],
             [-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1],
             [1,0,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,0],
             [1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,0],
             [-1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1],
             [-1,1,1,-1,-1,1,1,1,0,-1,1,1,-1,-1,1,1],
             [-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,1,1,-1,0],
             [-1,-1,1,-1,-1,1,1,1,-1,-1,1,-1,-1,1,0,1],
             [1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1],
             [1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,1,1],
             [1,-1,-1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1],
             [-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,1,-1,1],
             [-1,0,1,0,-1,1,1,1,1,1,1,-1,0,0,1,1],
             [-1,1,1,-1,1,0,1,-1,-1,1,1,-1,1,-1,1,1],
             [-1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,-1,1,-1,1],
             [-1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1],
             [-1,-1,-1,1,1,1,1,-1,-1,1,-1,1,-1,1,1,1],
             [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
             [1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1]]


votos_valid_clas=['republicano',
                  'democrata',  
                  'republicano',
                  'republicano',
                  'republicano',
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'republicano',
                  'republicano',
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'republicano',
                  'democrata',  
                  'republicano',
                  'republicano',
                  'republicano',
                  'republicano',
                  'republicano',
                  'democrata',  
                  'republicano',
                  'democrata',  
                  'republicano',
                  'democrata',  
                  'democrata',  
                  'republicano',
                  'republicano',
                  'republicano',
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'republicano',
                  'democrata',  
                  'democrata',  
                  'republicano',
                  'democrata',  
                  'democrata',  
                  'republicano',
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'republicano',
                  'democrata',  
                  'democrata',  
                  'democrata',  
                  'republicano',
                  'republicano',
                  'democrata',  
                  'democrata',  
                  'republicano',
                  'democrata',  
                  'republicano',
                  'republicano',
                  'republicano']



votos_test=[[1,-1,1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,1,0],
            [-1,1,1,1,1,1,1,1,1,-1,-1,1,1,1,-1,1],
            [-1,1,-1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,1,1],
            [-1,-1,1,1,1,1,1,1,1,1,-1,1,1,1,1,1],
            [-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,-1,1,0],
            [-1,-1,1,1,1,1,1,-1,-1,1,1,1,1,1,-1,1],
            [-1,1,1,-1,-1,1,1,1,1,1,-1,0,-1,-1,1,1],
            [1,-1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
            [1,1,1,-1,-1,1,1,1,1,1,1,1,1,1,-1,0],
            [-1,-1,-1,1,1,1,-1,-1,-1,1,0,1,1,1,-1,1],
            [1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,1],
            [1,-1,1,-1,1,1,1,-1,1,1,-1,-1,1,1,-1,0],
            [1,1,1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,1],
            [1,1,-1,1,1,1,-1,-1,-1,1,1,-1,1,-1,-1,-1],
            [1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,1],
            [-1,1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,1,-1,-1],
            [1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1,-1,0],
            [1,1,1,-1,1,1,1,1,-1,1,1,-1,-1,-1,1,0],
            [-1,1,1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,-1,1],
            [1,1,1,-1,0,1,1,1,-1,1,0,0,-1,-1,1,1],
            [1,1,1,-1,0,-1,1,1,1,1,-1,-1,-1,-1,1,0],
            [-1,1,1,1,1,1,-1,-1,-1,-1,1,1,0,1,-1,-1],
            [-1,1,1,0,1,1,-1,1,-1,1,0,-1,1,1,0,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1],
            [1,0,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1],
            [-1,1,-1,1,1,1,0,0,-1,-1,0,0,1,0,0,0],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1],
            [1,1,1,-1,-1,1,0,1,1,-1,1,-1,1,-1,1,1],
            [1,1,1,-1,1,1,1,1,1,1,1,-1,1,1,-1,0],
            [1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,0],
            [1,1,1,-1,1,1,-1,1,1,1,1,-1,-1,-1,-1,1],
            [1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1],
            [1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,1,1,1,-1],
            [-1,0,1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,0],
            [1,1,1,-1,1,1,-1,1,1,-1,1,-1,-1,1,-1,0],
            [-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,0],
            [1,-1,1,-1,-1,-1,1,1,1,0,1,-1,-1,-1,1,0],
            [0,0,-1,-1,0,1,0,-1,-1,-1,1,1,-1,1,-1,0],
            [1,1,-1,-1,-1,-1,-1,1,1,-1,1,-1,-1,-1,1,-1],
            [1,1,-1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1],
            [0,0,0,0,-1,1,-1,1,1,-1,-1,1,1,-1,-1,0],
            [1,1,0,0,0,1,-1,-1,-1,-1,1,-1,1,-1,-1,1],
            [1,1,1,0,-1,-1,-1,1,-1,-1,1,0,-1,-1,1,1],
            [1,1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,1,1],
            [1,1,-1,-1,1,0,-1,-1,-1,-1,1,-1,1,1,-1,1],
            [-1,1,1,-1,1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,1],
            [-1,1,-1,1,0,1,-1,-1,-1,1,-1,1,1,1,-1,-1],
            [-1,1,-1,1,1,1,-1,0,-1,-1,0,0,0,1,-1,0],
            [-1,1,-1,1,1,1,-1,-1,-1,1,1,1,1,1,-1,-1],
            [0,-1,1,1,-1,1,1,1,1,1,-1,1,-1,1,-1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,0,1,-1,-1],
            [1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,1],
            [1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1],
            [1,-1,1,-1,-1,1,1,1,1,-1,-1,1,0,1,1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1],
            [-1,-1,-1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1],
            [1,-1,1,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,1],
            [-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1],
            [1,1,1,1,1,1,1,1,-1,1,0,0,0,1,-1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1],
            [-1,1,1,-1,-1,1,1,1,0,1,-1,-1,-1,-1,-1,1],
            [1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,1,1,-1,1],
            [1,1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,1],
            [1,1,1,-1,-1,-1,1,1,-1,1,-1,-1,-1,-1,-1,1],
            [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1],
            [1,1,1,1,1,1,1,1,-1,1,-1,-1,1,1,-1,1],
            [-1,1,1,-1,1,1,1,1,-1,-1,1,-1,1,-1,1,1],
            [-1,-1,1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,1,1],
            [-1,1,1,-1,-1,1,1,1,1,-1,1,-1,-1,1,1,1],
            [-1,1,1,-1,-1,0,1,1,1,1,1,-1,0,1,1,1],
            [-1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1,1,0],
            [1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1],
            [-1,-1,-1,1,1,1,1,1,-1,1,-1,1,1,1,-1,1],
            [0,0,0,-1,-1,-1,1,1,1,1,-1,-1,1,-1,1,1],
            [1,-1,1,-1,0,-1,1,1,1,1,-1,1,-1,0,1,1],
            [-1,-1,1,1,1,1,-1,-1,1,1,-1,1,1,1,-1,1],
            [-1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1],
            [-1,0,-1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1],
            [-1,-1,-1,1,1,1,0,0,0,0,-1,1,1,1,-1,1],
            [-1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,0,-1]]
            
                  
                  
                  
                  
votos_test_clas=['democrata',  
                 'republicano',                  
                 'democrata',                    
                 'republicano',                  
                 'democrata',                    
                 'republicano',                  
                 'democrata',                    
                 'republicano',                  
                 'republicano',                  
                 'republicano',                  
                 'democrata',                    
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'republicano',
                 'republicano',
                 'republicano',
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'republicano',
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'democrata',  
                 'republicano',
                 'democrata',  
                 'republicano',
                 'republicano',
                 'republicano']
