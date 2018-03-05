# -*- coding: utf-8 -*-

import random
from clasificadores.clasificador import Clasificador

class Maximizar(Clasificador):

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
        return calcular_prediccion(ej, self.pesos, None)

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
        pesos = pesos_iniciales
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