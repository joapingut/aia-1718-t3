# -*- coding: utf-8 -*-
import copy

import clasificadores.perceptron as Perceptron
import clasificadores.clasificador as Clasificador
import random, math


'''
Metodo que genera un conjunto aleatorio de ejemplos.
Los parametros son:
    rango: rango en el que se mueven los valores de los atributos (numeros enteros)
    dim: numero de atributos
    tam: numero de ejemplos que se desean
    separable: booleano que indica si el conjunto es linealmente separable o no
    clases: lista que indica el nombre de las clases. Si se pasa None lso atributos seran valores desde 0 al numero de atributos
    pesos: lista de pesos para generar el hiperplano por si se desea reutilizar una ejecucion anterior

El resultado del metodo es un lista con 3 elementos
    indice 0: lista con el valor del hiperplano
    indice 1: lista con los ejemplos del conjunto
    indice 2: lista con el resultado esperado para cada elemento del indice 1
'''
def generar_conjunto_aleatorio(rango, dim, tam, separable=True, clases=None, pesos=None):
    hiperplano = None
    if pesos is None:
        hiperplano = Clasificador.genera_pesos(dim)
    else:
        hiperplano = pesos
    conjunto = []
    soluciones = []
    for i in range(0, tam):
        atributos = []
        for j in range(0, dim):
            atributos.append(random.randint(-rango, rango))
        prediccion = Clasificador.calcular_prediccion(atributos, hiperplano, None, is_sigma=False)
        conjunto.append(atributos)
        if clases != None:
            soluciones.append(clases[prediccion])
        else:
            soluciones.append(prediccion)
    if not separable:
        soluciones = altera_soluciones(soluciones, rango)
    return (hiperplano, conjunto, soluciones)

'''
Metodo que dado una lista con las soluciones de los ejemplos altera un  porcentaje de ellos.
Esto se hace para converir un conjunto linealmente separable en otro que no lo es.
Los resultados se van escogiendo aleatoriamente.
'''
def altera_soluciones(originales, rango, porcentaje=0.2):
    soluciones = copy.copy(originales)
    percent = math.floor(len(originales) * porcentaje)
    used = []
    for i in range(0, percent):
        index = random.randint(0, len(originales) -1)
        while index in used:
            index = random.randint(0, len(originales) -1)
        used.append(index)
        valor = random.randint(-rango, rango)
        while valor == originales[index]:
            valor = random.randint(-rango, rango)
        soluciones[index] = valor
    return soluciones