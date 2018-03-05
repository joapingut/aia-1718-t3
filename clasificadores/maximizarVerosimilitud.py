# -*- coding: utf-8 -*-

import random
from clasificadores.clasificador import Clasificador
import clasificadores.clasificador as clasificador

class Maximizar(Clasificador):

    def __init__(self, clases, norm=False, estocastico=True):
        self.clases = clases
        self.normaliza = norm
        self.entrenado = False
        self.pesos = None
        self.estocastico = estocastico
        None

    def entrena(self, entr, clas_entr, n_epochs, rate=0.1, pesos_iniciales=None, rate_decay=False):
        self.pesos = entrena(entr, clas_entr, self.clases, n_epochs, rate, pesos_iniciales, rate_decay, self.estocastico)
        self.entrenado = True
        None

    def clasifica_prob(self, ej):
        return clasificador.calcular_prediccion(ej, self.pesos, None, is_sigma=True)

    def clasifica(self, ej):
        return clasificador.calcular_prediccion(ej, self.pesos, self.clases, is_sigma=True)

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


def entrena(conjunto, resultados, clases, n_epochs, rate_inicial, pesos_iniciales, rate_decay, estocastico=True):
    pesos = None
    rate = rate_inicial
    if pesos_iniciales == None:
        pesos = clasificador.genera_pesos(len(conjunto[0]))
    else:
        pesos = pesos_iniciales
    epoch = 0
    n_errors = 1
    while epoch < n_epochs or n_errors == 0:
        n_errors = 0
        for index in range(0, len(conjunto)):
            prediccion = clasificador.calcular_prediccion(conjunto[index], pesos, clases, is_sigma=True)
            if prediccion != resultados[index]:
                n_errors += 1
                if estocastico:
                    pesos = ajusta_pesos_estocastico(conjunto[index], pesos, clasificador.busca_resultado(resultados[index], clases), rate)
        if not estocastico:
                    pesos = ajusta_pesos_batch(conjunto, pesos, resultados, clases, rate)
        if rate_decay:
            rate = clasificador.decaer_ratio(rate, epoch)
        epoch += 1
    return pesos

def ajusta_pesos_estocastico(conjunto, pesos, esperado, rate):
    coef = []
    coef.append(pesos[0] + rate * (esperado - clasificador.umbral(pesos[0])))
    for i in range(0, len(conjunto)):
        coef.append(pesos[i + 1] + rate * conjunto[i] * (esperado - clasificador.sigma(clasificador.calcular_producto_escalar(pesos, conjunto))))
    return coef