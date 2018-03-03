# -*- coding: utf-8 -*-

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