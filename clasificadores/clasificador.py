# -*- coding: utf-8 -*-

import random, copy, math, numpy

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
    mean = []
    desviacion = []
    num_attr = len(conjunto[0])
    nun_arra = numpy.asarray(conjunto)
    for j in range(0, num_attr):
        mean.append(numpy.mean(nun_arra[:,j]))
        desviacion.append(numpy.std(nun_arra[:,j]))
    return (mean, desviacion)

def normalizar(ejemplo, media, desviacion):
    num_attr = len(ejemplo[0])
    norm_arra = []
    for i in range(0, len(ejemplo)):
        n_element = []
        for j in range(0, num_attr):
            num = ejemplo[i][j]
            nnum = puntuacion_estandar(num,media[j],desviacion[j])
            n_element.append(nnum)
        norm_arra.append(n_element)
    return norm_arra

def normalizar_elemento(ejemplo, media, desviacion):
    norm_arra = []
    for j in range(0, len(ejemplo)):
        num = ejemplo[j]
        nnum = puntuacion_estandar(num,media[j],desviacion[j])
        norm_arra.append(nnum)
    return norm_arra

def puntuacion_estandar(num, media, desvia):
    return (num - media) / desvia