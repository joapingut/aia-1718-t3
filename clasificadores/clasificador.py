# -*- coding: utf-8 -*-

import random, copy, math

'''
Clasificador es la clase básica que describe un clasificador sobre la que se implementaran el resto de clasificadores.
'''
class Clasificador():

    def __init__(self, clases, norm=False):
        None

    def entrena(self, entr, clas_entr, n_epochs, rate=0.1, pesos_iniciales=None, rate_decay=False):
        None

    def clasifica_prob(self, ej):
        None

    def clasifica(self, ej):
        None

    def evalua(self, validacion):
        None

    def imprime(self):
        None

'''
Clase que lanza una excepción. Usada para cuando un clasificador no esta entrenado aún.
'''
class ClasificadorNoEntrenado(Exception): pass


def calcular_prediccion(conjunto, pesos, clases, is_sigma=False):
    coef = calcular_producto_escalar(pesos, conjunto)
    if not is_sigma:
        result = umbral(coef)
    else:
        result = sigma(coef)
    if clases != None:
        if is_sigma:
            result = int(round(result))
        return clases[result]
    return result

def decaer_ratio(rate, epoch):
    #return rate + 2/(((epoch + 1) ** 2) **(1.0/3.0))
    return rate + 2/(math.pow(epoch + 1, 2/3))


def busca_resultado(busco, clases):
    for index in range(0, len(clases)):
        if clases[index] == busco:
            return index
    return None

def calcular_producto_escalar(pesos, atributos):
    coef = pesos[0]
    for i in range(0, len(atributos)):
        coef += pesos[i + 1] * atributos[i]
    return coef

def umbral(num):
    if num >= 0:
        return 1
    return 0

def sigma(z):
    try:
        return 1/(1 + math.exp(-z))
    except OverflowError:
        return 1

'''
    Genera una lista de pesos aleatorios.
    El elemento 0 de la lista es siempre el peso del
    termino independiente que no entra dentro del
    parametro, la lista final siempre tiene num + 1
    elementos
'''
def genera_pesos(num):
    result = []
    for i in range(0, num + 1):
        result.append(random.uniform(-1,1))
    return result

def extrae_normalizacion(conjunto):
    for i in range(0, len(conjunto)):
        None
    None

def normalizar(ejemplo, media, desviacion):
    None