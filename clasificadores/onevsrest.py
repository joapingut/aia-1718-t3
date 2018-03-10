# -*- coding: utf-8 -*-

import clasificadores.clasificador as clasificador
import clasificadores.perceptron as perceptron
import clasificadores.regresionLogistica as regresion
import clasificadores.maximizarVerosimilitud as maximizar

class One_vs_Rest():

    def __init__(self, clases, clasificador, estocastico=True, norm=False):
        self.clases = clases
        self.clasificador = clasificador
        self.clasificadores = None
        self.estocastico = estocastico
        self.normaliza = norm
        self.entrenado = False

    def entrena(self, entr, clas_entr, n_epochs, rate=0.1, rate_decay=False):
        clasificadores = []
        for clase in self.clases:
            clases = ['other', clase]
            resultados = replace_clases(clase, 'other', clas_entr)
            clasificador = obtiene_clasificador(self.clasificador, clases, self.estocastico, self.normaliza)
            clasificador.entrena(entr, resultados, n_epochs, rate=rate, rate_decay=rate_decay)
            clasificadores.append(clasificador)
        self.entrenado = True
        self.clasificadores = clasificadores

    def clasifica_prob(self, ej):
        mejor = None
        mejor_i = None
        for i in range(0, len(self.clasificadores)):
            clasificador = self.clasificadores[i]
            res = clasificador.clasifica_prob(ej)
            if mejor == None or res > mejor:
                mejor = res
                mejor_i = i
        return (self.clases[mejor_i], mejor)

    def clasifica(self, ej):
        mejor = None
        mejor_i = None
        for i in range(0, len(self.clasificadores)):
            clasificador = self.clasificadores[i]
            res = clasificador.clasifica_prob(ej)
            if mejor == None or res > mejor:
                mejor = res
                mejor_i = i
        return self.clases[mejor_i]

    def evalua(self, validacion, resultados):
        exito = 0
        for index in range(0, len(validacion)):
            predicho = self.clasifica(validacion[index])
            if (resultados[index] == predicho):
                exito += 1
        return exito/len(validacion)

    def imprime(self):
        None

def replace_clases(mantener, otros, original):
    result = []
    for elemento in original:
        if elemento == mantener:
            result.append(elemento)
        else:
            result.append(otros)
    return result

def obtiene_clasificador(tipo, clases, estocastico, normalizar):
    if tipo == 'perceptron':
        return perceptron.Perceptron(clases, norm=normalizar)
    if tipo == 'verosimilitud':
        return maximizar.Maximizar(clases, norm=normalizar, estocastico=estocastico)
    if tipo == 'regresion':
        return maximizar.Regresion(clases, norm=normalizar, estocastico=estocastico)

