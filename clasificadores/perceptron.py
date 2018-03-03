# -*- coding: utf-8 -*-

import random
from clasificadores.clasificador import Clasificador

__author__ = 'Joaquin'

class Perceptron(Clasificador):

    def __init__(self, clases, norm=False):
        self.clases = clases
        self.normaliza = norm
        self.entrenado = False
        self.pesos = None
        None

    def entrena(self, entr, clas_entr, n_epochs, rate=0.1, pesos_iniciales=None, rate_decay=False):
        self.pesos = entrena(entr, clas_entr, self.clases, n_epochs, rate, pesos_iniciales, rate_decay)
        self.entrenado = True
        None

    def clasifica_prob(self, ej):
        None

    def clasifica(self, ej):
        return calcular_prediccion(ej, self.pesos, self.clases)

    def evalua(self, validacion, resultados):
        exito = 0
        for index in range(0, len(validacion)):
            predicho = self.clasifica(validacion[index])
            if (resultados[index] == predicho):
                exito += 1
        return exito/len(validacion)

    def set_Pesos(self, pesos):
        self.pesos = pesos
        self.entrenado = True

    def imprime(self):
        return str(self.pesos)

def entrena(conjunto, resultados, clases, n_epochs, rate_inicial, pesos_iniciales, rate_decay):
    pesos = None
    rate = rate_inicial
    if pesos_iniciales == None:
        pesos = genera_pesos(len(conjunto[0]))
    else:
        pesos = pesos_iniciales + genera_pesos(0)
    epoch = 0
    n_errors = 1
    while epoch < n_epochs or n_errors == 0:
        n_errors = 0
        for index in range(0, len(conjunto)):
            prediccion = calcular_prediccion(conjunto[index], pesos, clases)
            if prediccion != resultados[index]:
                n_errors += 1
                pesos = ajusta_pesos(conjunto[index], pesos, busca_resultado(resultados[index], clases), rate)
        if rate_decay:
            rate = decaer_ratio(rate, epoch)
        epoch += 1
    return pesos

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

def calcular_prediccion(conjunto, pesos, clases):
    coef = []
    coef.append(pesos[0])
    for i in range(0, len(conjunto)):
        coef.append(pesos[i + 1] * conjunto[i])
    result = umbral(sum(coef))
    if clases != None:
        return clases[result]
    return result

def umbral(num):
    if num >= 0:
        return 1
    return 0

def ajusta_pesos(conjunto, pesos, esperado, rate):
    coef = []
    coef.append(pesos[0] + rate * (esperado - umbral(pesos[0])))
    for i in range(0, len(conjunto)):
        coef.append(pesos[i + 1] + rate * conjunto[i] * (esperado - umbral(pesos[i + 1] * conjunto[i])))
    return coef

def decaer_ratio(rate, epoch):
    return rate + 2/((epoch + 1) **(1/3))

def busca_resultado(busco, clases):
    for index in range(0, len(clases)):
        if clases[index] == busco:
            return index
    return None