# -*- coding: utf-8 -*-
import copy

import clasificadores.perceptron as Perceptron
import clasificadores.clasificador as Clasificador
import random, math

__author__ = 'Joaquin'


def generar_conjunto_aleatorio(rango, dim, tam, separable=True, clases=None):
    hiperplano = Clasificador.genera_pesos(dim)
    conjunto = []
    soluciones = []
    for i in range(0, tam):
        atributos = []
        for j in range(0, dim):
            atributos.append(random.randint(-rango, rango))
        prediccion = Perceptron.calcular_prediccion(atributos, hiperplano, None)
        conjunto.append(atributos)
        if clases != None:
            soluciones.append(clases[prediccion])
        else:
            soluciones.append(prediccion)
    if not separable:
        soluciones = altera_soluciones(soluciones, rango)
    return (hiperplano, conjunto, soluciones)

def altera_soluciones(originales, rango):
    soluciones = copy.copy(originales)
    percent = math.floor(len(originales) * 0.20)
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