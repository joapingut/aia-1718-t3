# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm, tree, ensemble
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


lbc = load_breast_cancer()

lbc_atributes, lbc_classnames = lbc.feature_names, lbc.target_names

'''Función proporcionada en las diapositivas del tema para representar
nubes de puntos en función de dos valores de atributo.'''

def representacion_grafica(datos,caracteristicas,objetivo,clases,c1,c2):

    for tipo,marca,color in zip(range(len(clases)),"soD","rgb"):
        plt.scatter(datos[objetivo == tipo,c1],datos[objetivo == tipo,c2],marker=marca,c=color)
    
    plt.xlabel(caracteristicas[c1])
    plt.ylabel(caracteristicas[c2])
    plt.legend(clases)
    plt.show()

'''Función para aplicar el clasificador Knn en el conjunto de datos. Se forman dos conjuntos:
uno de entrenamiento y otro de test para probar el rendimiento del modelo, esto se realizará
también en el resto de clasificadores de esta parte. Como parámetros de entrada están los
ejemplos del conjunto de datos y los valores de clasificación de dichos ejemplos, además de
un parámetro normalizar que normaliza los datos si entra un True, o no lo hace en caso de
False. Se aplican dos tipos de Knn dependiendo del valor que se le den a los pesos (por distancia
(tienen mayor peso los puntos más cercanos) o uniforme). Se devuelven los clasificadores
y los rendimientos en una lista para su posterior manipulación, esto se realiza también
en el resto de clasificadores.'''

def knn(datos=lbc.data, objetivos=lbc.target, normalizar=True, neighbors=7):

    examples_train, examples_test, classes_train, classes_test = train_test_split(datos, objetivos, test_size = 0.25)

    if normalizar:
        normalizadorTrain = StandardScaler().fit(examples_train)
        normalizadorTest = StandardScaler().fit(examples_test)

        examples_train = normalizadorTrain.transform(examples_train) 
        examples_test = normalizadorTest.transform(examples_test)

    knnDistancia = KNeighborsClassifier(n_neighbors=neighbors,weights='distance')
    knnUniforme = KNeighborsClassifier(n_neighbors=neighbors,weights='uniform')
    knnDistancia.fit(examples_train, classes_train)
    knnUniforme.fit(examples_train, classes_train)

    rendimientoDistancia = knnDistancia.score(examples_test, classes_test)
    rendimientoUniforme = knnUniforme.score(examples_test, classes_test)

    print("Para knn distancia:"+str(rendimientoDistancia))
    print("Para knn uniforme:"+str(rendimientoUniforme))
    
    return [knnDistancia, knnUniforme, rendimientoDistancia, rendimientoUniforme]

'''El siguiente clasificador ejecuta los clasificadores de perceptrón, regresión logística
y vectores soporte dependiendo del valor de entrada loss. Los parámetros que se han considerado
son los epoch (máximo número de iteraciones), penalty (tipo de regularización), valor de alpha
en caso de aplicarse dicha regularización, learning_rate (tasa de aprendizaje) y aplicar o no
dicha tasa.'''

def clasificadorSGD(datos=lbc.data, objetivos=lbc.target,epoch=100,regularizacion=None,alpha=0.01,tasa_aprendizaje=0.0,aplicar_tasa=False):
    tasa = "optimal"
    
    if aplicar_tasa:
        tasa = "constant"
    
    examples_train, examples_test, classes_train, classes_test = train_test_split(datos, objetivos, test_size = 0.25)

    csgdPerceptron = linear_model.SGDClassifier(loss="perceptron",max_iter=epoch,penalty=regularizacion,alpha=alpha,learning_rate=tasa,eta0=tasa_aprendizaje)
    csgdRegresionLogistica = linear_model.SGDClassifier(loss="log",max_iter=epoch,penalty=regularizacion,alpha=alpha,learning_rate=tasa,eta0=tasa_aprendizaje)
    csgdVectoresSoporte = linear_model.SGDClassifier(loss="hinge",max_iter=epoch,penalty=regularizacion,alpha=alpha,learning_rate=tasa,eta0=tasa_aprendizaje)

    
    csgdPerceptron.fit(examples_train, classes_train)
    csgdRegresionLogistica.fit(examples_train, classes_train)
    csgdVectoresSoporte.fit(examples_train, classes_train)
    
    rendimientoPerceptron = csgdPerceptron.score(examples_test, classes_test)
    rendimientoRegLog = csgdRegresionLogistica.score(examples_test, classes_test)
    rendimientoVectSop = csgdVectoresSoporte.score(examples_test, classes_test)
    
    print("Para perceptron:"+str(rendimientoPerceptron))
    print("Para regresion logistica:"+str(rendimientoRegLog))
    print("Para vectores soporte:"+str(rendimientoVectSop))
    
    return [csgdPerceptron, csgdRegresionLogistica, csgdVectoresSoporte, rendimientoPerceptron, rendimientoRegLog, rendimientoVectSop]
    
'''La siguiente función aplica el clasificador de regresión logística pero sin hacer uso de SGDC,
los parámetros de entrada que se consideran son los epoch, el tipo de regularización (l1, l2)
y el valor de C (a mayor C, menor tolerancia a aceptar puntos fuera de la región que separa
el hiperplano).'''

def regresionLogistica(datos=lbc.data, objetivos=lbc.target,epoch=100,regularizacion="l1",c=0.01):
    
    examples_train, examples_test, classes_train, classes_test = train_test_split(datos, objetivos, test_size = 0.25)

    regresionLogistica = linear_model.LogisticRegression(max_iter=epoch,penalty=regularizacion,C=c)
    
    regresionLogistica.fit(examples_train, classes_train)
    
    rendimientoRegLog = regresionLogistica.score(examples_test, classes_test)
    
    print("Para regresion logistica (no SGD):"+str(rendimientoRegLog))
    
    return [regresionLogistica,rendimientoRegLog]
    
'''La siguiente función aplica el clasificador de vectores soporte sin hacer uso de SGDC.
Los parámetros de entrada que se han considerado son el número de epoch, el valor de C, y
loss (para aplicar "hinge" o "squared hinge". Para l1 da problemas con ambos posibles 
valores de loss "hinge" y "hinge_squared", así que solo se usa l2.'''    
    
def vectoresSoporte(datos=lbc.data, objetivos=lbc.target,epoch=100,c=0.01, loss=False):
    
    examples_train, examples_test, classes_train, classes_test = train_test_split(datos, objetivos, test_size = 0.25)

    if loss:
        vectoresSoporte = svm.LinearSVC(max_iter=epoch,C=c, loss="hinge")
    else:
        vectoresSoporte = svm.LinearSVC(max_iter=epoch,C=c)
    
    vectoresSoporte.fit(examples_train, classes_train)
    
    rendimientoVectSop = vectoresSoporte.score(examples_test, classes_test)
    
    print("Para vectores soporte (no SGD):"+str(rendimientoVectSop))
    
    return [vectoresSoporte,rendimientoVectSop]
    
'''La siguiente función aplica el clasificador de árboles de decisión. Los parámetros que
se han considerado son el de profundidad del árbol para prepoda, el criterio a aplicar (gini
o entropia) y ejemplosMin (un nodo es candidato o no dependiendo de la proporción de ejemplos).'''

def arbolesDecision(datos=lbc.data, objetivos=lbc.target, profundidad=None, criterio="gini", ejemplosMin=0.1):
    
    examples_train, examples_test, classes_train, classes_test = train_test_split(datos, objetivos, test_size = 0.25)

    arbolDecision = tree.DecisionTreeClassifier(max_depth=profundidad,criterion=criterio, min_samples_split=ejemplosMin)

    arbol = arbolDecision.fit(examples_train, classes_train)
    
    rendimientoArbol = arbol.score(examples_test, classes_test)
    
    print("Arbol de decision:"+str(rendimientoArbol))
    
    return [arbol, rendimientoArbol]

'''La siguiente función aplica el clasificador de random forest. Los parámetros que
se han considerado son el de profundidad del árbol para prepoda, bootstrap (coger conjuntos
de ejemplos más pequeños respecto al grande), el criterio a aplicar (gini o entropia)
y ejemplosMin (un nodo es candidato o no dependiendo de la proporción de ejemplos).'''

def randomForest(datos=lbc.data, objetivos=lbc.target, profundidad=None, bootstrap=True, criterio="gini", ejemplosMin=2):
    
    examples_train, examples_test, classes_train, classes_test = train_test_split(datos, objetivos, test_size = 0.25)

    arboles = ensemble.RandomForestClassifier(max_depth=profundidad,bootstrap=bootstrap,criterion=criterio, min_samples_split=ejemplosMin)

    arboles = arboles.fit(examples_train, classes_train)
    
    rendimientoArboles = arboles.score(examples_test, classes_test)
    
    print("Random forest:"+str(rendimientoArboles))
    
    return [arboles,rendimientoArboles]

'''Las siguientes funciones son de pruebas para los distintos clasificadores, en las que
se inicializan los posibles valores de parámetros para cada uno y se recorren en bucles
escalonados, ejecutando dichos clasificadores y comprobando el rendimiento en cada iteración.
Finalmente, se devuelven los parámetros usados en la ejecución de mejor rendimiento y el
rendimiento.'''

def pruebasKnn(datos=lbc.data, objetivos=lbc.target):
    rendDist = 0.0
    rendUni = 0.0
    parametroNorm = [True, False]
    parametroNeighbors = [3,4,5,6,7,8,9]
    for norm in parametroNorm:
        for neighbors in parametroNeighbors:
            knnActual = knn(datos,objetivos,norm,neighbors)
            if knnActual[2] > rendDist:
                dist = [norm,neighbors,knnActual[0],knnActual[2]]
                rendDist = knnActual[2]
            if knnActual[3] > rendUni:
                uni = [norm,neighbors,knnActual[1],knnActual[3]]
                rendUni = knnActual[3]
    res = [dist,uni]
    return res

def pruebasSGDC(datos=lbc.data, objetivos=lbc.target,epoch=100,regularizacion=None,alpha=0.01,tasa_aprendizaje=0.0,aplicar_tasa=False):
    rendPerceptron = 0.0
    rendRegLog = 0.0
    rendVectSop = 0.0
    parametroEpoch = [10,100,500,1000,5000,10000]
    parametroPenalty = ["none","l1","l2"]
    parametroAlpha = [0.001,0.01,0.1,0.5,0.7]
    parametroTasaAp = [0.001,0.01,0.1,0.5]
    parametroAplicar = [True, False]
    for epoch in parametroEpoch:
        for penalty in parametroPenalty:
            for alpha in parametroAlpha:
                for tasa in parametroTasaAp:
                    for aplicar in parametroAplicar:
                        SGDCActual = clasificadorSGD(datos,objetivos,epoch,penalty,alpha,tasa,aplicar)
                        if SGDCActual[3] > rendPerceptron:
                            perceptron = [epoch,penalty,alpha,tasa,aplicar,SGDCActual[3]]
                            rendPerceptron = SGDCActual[3]
                        if SGDCActual[4] > rendRegLog:
                            reg = [epoch,penalty,alpha,tasa,aplicar,SGDCActual[4]]
                            rendRegLog = SGDCActual[4]
                        if SGDCActual[5] > rendVectSop:
                            vect = [epoch,penalty,alpha,tasa,aplicar,SGDCActual[5]]
                            rendVectSop = SGDCActual[5]
    res = [perceptron,reg,vect]
    return res

def pruebasRegresionLogistica(datos=lbc.data, objetivos=lbc.target):
    rendRegLog = 0.0
    parametroEpoch = [10,100,500,1000,5000,10000]
    parametroPenalty = ["l1","l2"]
    parametroC = [0.0001,0.001,0.005,0.01,0.1]
    for epoch in parametroEpoch:
        for penalty in parametroPenalty:
            for c in parametroC:
                RegLogActual = regresionLogistica(datos,objetivos,epoch,penalty,c)
                if RegLogActual[1] > rendRegLog:
                    regresionlog = [epoch,penalty,c,RegLogActual[1]]
                    rendRegLog = RegLogActual[1]
    return regresionlog

def pruebasVectoresSoporte(datos=lbc.data, objetivos=lbc.target):
    rendVectSop = 0.0
    parametroEpoch = [10,100,500,1000,5000,10000]
    parametroLoss = [True, False]
    parametroC = [0.0001,0.001,0.005,0.01,0.1]
    for epoch in parametroEpoch:
        for loss in parametroLoss:
            for c in parametroC:
                VectSopActual = vectoresSoporte(datos,objetivos,epoch,c,loss)
                if VectSopActual[1] > rendVectSop:
                    vectsop = [epoch,c,loss,VectSopActual[1]]
                    rendVectSop = VectSopActual[1]
    return vectsop

def pruebasArbolDecision(datos=lbc.data, objetivos=lbc.target):
    rendAD = 0.0
    parametroProfundidad = [2,3,4,5,6,7,8,9,10]
    parametroCriterio = ["gini", "entropy"]
    parametroMinEjemplos = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for profundidad in parametroProfundidad:
        for criterio in parametroCriterio:
            for minej in parametroMinEjemplos:
                ADActual = arbolesDecision(datos,objetivos,profundidad,criterio,minej)
                if ADActual[1] > rendAD:
                    ad = [profundidad,criterio,minej,ADActual[1]]
                    rendAD = ADActual[1]
    return ad

def pruebasRandomForest(datos=lbc.data, objetivos=lbc.target):
    rendRF = 0.0
    parametroProfundidad = [2,3,4,5,6,7,8,9,10]
    parametroBootstrap = [True, False]
    parametroCriterio = ["gini", "entropy"]
    parametroMinEjemplos = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for profundidad in parametroProfundidad:
        for criterio in parametroCriterio:
            for minej in parametroMinEjemplos:
                for bootstrap in parametroBootstrap:
                    RFActual = randomForest(datos,objetivos,profundidad,bootstrap,criterio,minej)
                    if RFActual[1] > rendRF:
                        rf = [profundidad,bootstrap,criterio,minej,RFActual[1]]
                        rendRF = RFActual[1]
    return rf