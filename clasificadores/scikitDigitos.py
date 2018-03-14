# -*- coding: utf-8 -*-
import numpy as np
from procesamientoDigitos import procesarDigitosEscritos, procesarDigitos
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, svm, tree, ensemble

'''
Cargar los posibles valores de las clases guardándolos en un array de numpy.
'''

classes = ['0','1','2','3','4','5','6','7','8','9']
dd_classnames = np.array(classes)

'''
Cargar los conjuntos de entrenamiento y test a partir de los digitos escritos
y los valores de clase, guardándolos en arrays de numpy con 100 digitos para
entrenamiento y 25 para test.
'''

dd_examples_train = np.array(procesarDigitosEscritos("../datos/digitdata/trainingimages", limit=100))
dd_classes_train = np.array(procesarDigitos("../datos/digitdata/traininglabels", limit=100))

dd_examples_test = np.array(procesarDigitosEscritos("../datos/digitdata/trainingimages", limit=25))
dd_classes_test = np.array(procesarDigitos("../datos/digitdata/traininglabels", limit=25))

'''Todas las funciones siguientes son exactamente iguales a las de scikit.py adaptando los parámetros
de entrada para poder pasar los conjuntos de entrenamiento y prueba, además de modificaciones de los
valores en algunos parámetros.'''

def knn(examples_train=dd_examples_train, classes_train=dd_classes_train, examples_test=dd_examples_test, classes_test=dd_classes_test, neighbors=7):

    knnDistancia = KNeighborsClassifier(n_neighbors=neighbors,weights='distance')
    knnUniforme = KNeighborsClassifier(n_neighbors=neighbors,weights='uniform')
    knnDistancia.fit(examples_train, classes_train)
    knnUniforme.fit(examples_train, classes_train)

    rendimientoDistancia = knnDistancia.score(examples_test, classes_test)
    rendimientoUniforme = knnUniforme.score(examples_test, classes_test)

    print("Para knn distancia:"+str(rendimientoDistancia))
    print("Para knn uniforme:"+str(rendimientoUniforme))
    
    return [knnDistancia, knnUniforme, rendimientoDistancia, rendimientoUniforme]



def clasificadorSGD(examples_train=dd_examples_train, classes_train=dd_classes_train, examples_test=dd_examples_test, classes_test=dd_classes_test,epoch=100,regularizacion=None,alpha=0.01,tasa_aprendizaje=0.0,aplicar_tasa=False):
    tasa = "optimal"
    
    if aplicar_tasa:
        tasa = "constant"
    
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


def regresionLogistica(examples_train=dd_examples_train, classes_train=dd_classes_train, examples_test=dd_examples_test, classes_test=dd_classes_test,epoch=100,regularizacion="l1",c=0.01):
    
    regresionLogistica = linear_model.LogisticRegression(max_iter=epoch,penalty=regularizacion,C=c)
    
    regresionLogistica.fit(examples_train, classes_train)
    
    rendimientoRegLog = regresionLogistica.score(examples_test, classes_test)
    
    print("Para regresion logistica (no SGD):"+str(rendimientoRegLog))
    
    return [regresionLogistica,rendimientoRegLog]


def vectoresSoporte(examples_train=dd_examples_train, classes_train=dd_classes_train, examples_test=dd_examples_test, classes_test=dd_classes_test,epoch=100,c=0.01, loss=False):
    
    if loss:
        vectoresSoporte = svm.LinearSVC(max_iter=epoch,C=c, loss="hinge")
    else:
        vectoresSoporte = svm.LinearSVC(max_iter=epoch,C=c)
    
    vectoresSoporte.fit(examples_train, classes_train)
    
    rendimientoVectSop = vectoresSoporte.score(examples_test, classes_test)
    
    print("Para vectores soporte (no SGD):"+str(rendimientoVectSop))
    
    return [vectoresSoporte,rendimientoVectSop]


def arbolesDecision(examples_train=dd_examples_train, classes_train=dd_classes_train, examples_test=dd_examples_test, classes_test=dd_classes_test, profundidad=None, criterio="gini", ejemplosMin=0.1):
    
    arbolDecision = tree.DecisionTreeClassifier(max_depth=profundidad,criterion=criterio, min_samples_split=ejemplosMin)

    arbol = arbolDecision.fit(examples_train, classes_train)
    
    rendimientoArbol = arbol.score(examples_test, classes_test)
    
    print("Arbol de decision:"+str(rendimientoArbol))
    
    return [arbol, rendimientoArbol]

def randomForest(examples_train=dd_examples_train, classes_train=dd_classes_train, examples_test=dd_examples_test, classes_test=dd_classes_test, profundidad=None, bootstrap=True, criterio="gini", ejemplosMin=2):
    
    arboles = ensemble.RandomForestClassifier(max_depth=profundidad,bootstrap=bootstrap,criterion=criterio, min_samples_split=ejemplosMin)

    arboles = arboles.fit(examples_train, classes_train)
    
    rendimientoArboles = arboles.score(examples_test, classes_test)
    
    print("Random forest:"+str(rendimientoArboles))
    
    return [arboles,rendimientoArboles]



def pruebasKnn(examples_train=dd_examples_train, classes_train=dd_classes_train, examples_test=dd_examples_test, classes_test=dd_classes_test):
    rendDist = 0.0
    rendUni = 0.0
    parametroNeighbors = [3,4,5,6,7,8,9]
    for neighbors in parametroNeighbors:
        knnActual = knn(examples_train,classes_train,examples_test,classes_test,neighbors)
        if knnActual[2] > rendDist:
            dist = [neighbors,knnActual[0],knnActual[2]]
            rendDist = knnActual[2]
        if knnActual[3] > rendUni:
            uni = [neighbors,knnActual[1],knnActual[3]]
            rendUni = knnActual[3]
    res = [dist,uni]
    return res

def pruebasSGDC(examples_train=dd_examples_train, classes_train=dd_classes_train, examples_test=dd_examples_test, classes_test=dd_classes_test):
    rendPerceptron = 0.0
    rendRegLog = 0.0
    rendVectSop = 0.0
    parametroEpoch = [10,100,500,1000]
    parametroPenalty = ["none","l1","l2"]
    parametroAlpha = [0.001,0.01,0.1,0.5,0.7]
    parametroTasaAp = [0.001,0.01,0.1,0.5]
    parametroAplicar = [True, False]
    for epoch in parametroEpoch:
        for penalty in parametroPenalty:
            for alpha in parametroAlpha:
                for tasa in parametroTasaAp:
                    for aplicar in parametroAplicar:
                        SGDCActual = clasificadorSGD(examples_train,classes_train,examples_test,classes_test,epoch,penalty,alpha,tasa,aplicar)
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

def pruebasRegresionLogistica(examples_train=dd_examples_train, classes_train=dd_classes_train, examples_test=dd_examples_test, classes_test=dd_classes_test):
    rendRegLog = 0.0
    parametroEpoch = [10,100,500,1000]
    parametroPenalty = ["l1","l2"]
    parametroC = [0.0001,0.001,0.005,0.01,0.1]
    for epoch in parametroEpoch:
        for penalty in parametroPenalty:
            for c in parametroC:
                RegLogActual = regresionLogistica(examples_train,classes_train,examples_test,classes_test,epoch,penalty,c)
                if RegLogActual[1] > rendRegLog:
                    regresionlog = [epoch,penalty,c,RegLogActual[1]]
                    rendRegLog = RegLogActual[1]
    return regresionlog

def pruebasVectoresSoporte(examples_train=dd_examples_train, classes_train=dd_classes_train, examples_test=dd_examples_test, classes_test=dd_classes_test):
    rendVectSop = 0.0
    parametroEpoch = [10,100,500,1000]
    parametroLoss = [True, False]
    parametroC = [0.0001,0.001,0.005,0.01,0.1]
    for epoch in parametroEpoch:
        for loss in parametroLoss:
            for c in parametroC:
                VectSopActual = vectoresSoporte(examples_train,classes_train,examples_test,classes_test,epoch,c,loss)
                if VectSopActual[1] > rendVectSop:
                    vectsop = [epoch,c,loss,VectSopActual[1]]
                    rendVectSop = VectSopActual[1]
    return vectsop

def pruebasArbolDecision(examples_train=dd_examples_train, classes_train=dd_classes_train, examples_test=dd_examples_test, classes_test=dd_classes_test):
    rendAD = 0.0
    parametroProfundidad = [2,3,4,5,6,7,8,9,10]
    parametroCriterio = ["gini", "entropy"]
    parametroMinEjemplos = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for profundidad in parametroProfundidad:
        for criterio in parametroCriterio:
            for minej in parametroMinEjemplos:
                ADActual = arbolesDecision(examples_train,classes_train,examples_test,classes_test,profundidad,criterio,minej)
                if ADActual[1] > rendAD:
                    ad = [profundidad,criterio,minej,ADActual[1]]
                    rendAD = ADActual[1]
    return ad

def pruebasRandomForest(examples_train=dd_examples_train, classes_train=dd_classes_train, examples_test=dd_examples_test, classes_test=dd_classes_test):
    rendRF = 0.0
    parametroProfundidad = [2,3,4,5,6,7,8,9,10]
    parametroBootstrap = [True, False]
    parametroCriterio = ["gini", "entropy"]
    parametroMinEjemplos = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for profundidad in parametroProfundidad:
        for criterio in parametroCriterio:
            for minej in parametroMinEjemplos:
                for bootstrap in parametroBootstrap:
                    RFActual = randomForest(examples_train,classes_train,examples_test,classes_test,profundidad,bootstrap,criterio,minej)
                    if RFActual[1] > rendRF:
                        rf = [profundidad,bootstrap,criterio,minej,RFActual[1]]
                        rendRF = RFActual[1]
    return rf