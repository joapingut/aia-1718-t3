# -*- coding: utf-8 -*-

import clasificadores.perceptron as perceptron
import clasificadores.regresionLogistica as regresion
import clasificadores.maximizarVerosimilitud as maximizar

'''
Clase que implementa la clasificacion one vs rest
'''
class One_vs_Rest():

    '''
    Constructor del one vs rest.
        clases: las posibles clases que queremos clasificar
        clasificaro: cadena de texto con el clasificador que queremos usar, pueden ser perceptron, verosimilitud o regresion
        estocastico: si queremos la version estocastica o la batch
        norm: si queremos normalizar o no
    '''
    def __init__(self, clases, clasificador, estocastico=True, norm=False):
        self.clases = clases
        self.clasificador = clasificador
        self.clasificadores = None
        self.estocastico = estocastico
        self.normaliza = norm
        self.entrenado = False

    '''
    Metodo para entrenar el clasificador.
        entr: conjunto de entrenamiento
        clas_entr: resultado esperado del conjunto de entrenamiento
        n_epochs: numero de iteraciones a realizar
        rate: el ratio de aprendizaje
        pesos_iniciales: lista de pesos iniciales por si se desea retomar el entrenamiento
        rate_decay: indica si el ratio debe decaer con el tiempo o mantenerse igual.
    '''
    def entrena(self, entr, clas_entr, n_epochs, rate=0.1, rate_decay=False):
        clasificadores = []
        for clase in self.clases:
            clases = ['other', clase]
            print('Clase ', clase)
            resultados = replace_clases(clase, 'other', clas_entr)
            clasificador = obtiene_clasificador(self.clasificador, clases, self.estocastico, self.normaliza)
            clasificador.summary = None
            clasificador.entrena(entr, resultados, n_epochs, rate=rate, rate_decay=rate_decay)
            clasificadores.append(clasificador)
        self.entrenado = True
        self.clasificadores = clasificadores

    '''
    Metodo que dado un ejemplo indica la probabilidad de que tiene de pertenecer a una clase y otra.
    Valores entre 0 y 1 que si es menor de 0.5 indica que se acerca a la clase en la posicion 0
    de la lista de clases y si es mayor de 0.5 indica que se acerca a la clase en la posicion 1.
    Un valor de 0.5 indica indeterminacion pero por defecto se devuelve la clase en la posicion 0
    '''
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

    '''
    Metodo que clasifica un ejemplo y devuelve la clase en la que es mas probable que este se encuentre
    segun el entrenamiento que ha recibido el clasificador
    '''
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

    '''
    Metodo que evalua la precision del clasificador devolviendo un valor entre 0 y 1 que indica el
    porcentaje de acierto sobre el conjutno de validacion y sus resultados esperados que se pasan
    como parametros.
    '''
    def evalua(self, validacion, resultados):
        exito = 0
        for index in range(0, len(validacion)):
            predicho = self.clasifica(validacion[index])
            if (resultados[index] == predicho):
                exito += 1
        return exito/len(validacion)


'''
Metodo que reemplaza las clases de una lista de resultados.
    mantener: clase que no deseamos cambiar
    otros: nombre que recibiran el resto de clases
    original: lista de resultados original
'''
def replace_clases(mantener, otros, original):
    result = []
    for elemento in original:
        if elemento == mantener:
            result.append(elemento)
        else:
            result.append(otros)
    return result

'''
Metodo que devuelve un clasificador sin entrenar segun los parametros pasados
    tipo: tipo de clasidicador que deseamos
    clases: posibles clases entre las que clasificar. Solo puede tener dos elementos
    estocastico: si queremos la version estocastica o la batch
    normalizar: si queremos normalizar o no
'''
def obtiene_clasificador(tipo, clases, estocastico, normalizar):
    if tipo == 'perceptron':
        return perceptron.Perceptron(clases, norm=normalizar)
    if tipo == 'verosimilitud':
        return maximizar.Maximizar(clases, norm=normalizar, estocastico=estocastico)
    if tipo == 'regresion':
        return regresion.Regresion(clases, norm=normalizar, estocastico=estocastico)

