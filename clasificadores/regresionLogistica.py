# -*- coding: utf-8 -*-

import copy
from clasificadores.clasificador import Clasificador
import clasificadores.clasificador as clasificador

'''
Clase que implementa un clasificador por medio del metodo de la regresion logistica
'''
class Regresion(Clasificador):

    '''
    Constructor, se le debe pasar las clases.
    Opcionalmente se puede indicar si se quiere normalizar los datos y si se desea
    usar el metodo estocastico o no. Si esto ultimo es False se usara el metodo batch
    '''
    def __init__(self, clases, norm=False, estocastico=True):
        self.clases = clases
        self.normaliza = norm
        self.entrenado = False
        self.pesos = None
        self.estocastico = estocastico
        self.media = None
        self.desviacion = None
        self.summary = clasificador.Summary("Regresion logistica, estocastico: " + str(estocastico))
        None

    '''
    Metodo para entrenar el clasificador.
        entr: conjunto de entrenamiento
        clas_entr: resultado esperado del conjunto de entrenamiento
        n_epochs: numero de iteraciones a realizar
        rate: el ratio de aprendizaje
        pesos_iniciales: lista de pesos iniciales por si se desea retomar el entrenamiento
        rate_decay: indica si el ratio debe decaer con el tiempo o mantenerse igual.
    '''
    def entrena(self, entr, clas_entr, n_epochs, rate=0.1, pesos_iniciales=None, rate_decay=False):
        conjunto = entr
        if self.normaliza:
            self.media, self.desviacion = clasificador.extrae_normalizacion(entr)
            conjunto = clasificador.normalizar(entr, self.media, self.desviacion)
        self.pesos = entrena(conjunto, clas_entr, self.clases, n_epochs, rate, pesos_iniciales, rate_decay, self.estocastico, self.summary)
        self.entrenado = True
        None

    '''
    Metodo que dado un ejemplo indica la probabilidad de que tiene de pertenecer a una clase y otra.
    Valores entre 0 y 1 que si es menor de 0.5 indica que se acerca a la clase en la posicion 0
    de la lista de clases y si es mayor de 0.5 indica que se acerca a la clase en la posicion 1.
    Un valor de 0.5 indica indeterminacion pero por defecto se devuelve la clase en la posicion 0
    '''
    def clasifica_prob(self, ej):
        conjunto = ej
        if self.normaliza:
            conjunto = clasificador.normalizar_elemento(ej, self.media, self.desviacion)
        return clasificador.calcular_prediccion(conjunto, self.pesos, None, is_sigma=False)

    '''
    Metodo que clasifica un ejemplo y devuelve la clase en la que es mas probable que este se encuentre
    segun el entrenamiento que ha recibido el clasificador
    '''
    def clasifica(self, ej):
        conjunto = ej
        if self.normaliza:
            conjunto = clasificador.normalizar_elemento(ej, self.media, self.desviacion)
        return clasificador.calcular_prediccion(conjunto, self.pesos, self.clases, is_sigma=False)

    '''
    Metodo que evalua la precision del clasificador devolviendo un valor entre 0 y 1 que indica el
    porcentaje de acierto sobre el conjutno de validacion y sus resultados esperados que se pasan
    como parametros.
    '''
    def evalua(self, validacion, resultados):
        conjunto_val = validacion
        if self.normaliza:
            conjunto_val = clasificador.normalizar(validacion, self.media, self.desviacion)
        exito = 0
        for index in range(0, len(conjunto_val)):
            predicho = self.clasifica(conjunto_val[index])
            if (resultados[index] == predicho):
                exito += 1
        return exito/len(conjunto_val)

    '''
    Metodo para modificar los pesos del algoritmo sin tener que volver a entrenarlo
    '''
    def set_Pesos(self, pesos):
        self.pesos = pesos
        self.entrenado = True

    '''
    Metodo que imprime la lista de pesos actual del algoritmo
    '''
    def imprime(self):
        return str(self.pesos)


'''
Metodo para ralizar el entrenamiento del algoritmo.
    conjunto: conjunto de entrenamiento
    resultados: resultado esperado del conjunto de entrenamiento
    clases: posibles clases que pueden tomar los valores
    n_epochs: numero de iteraciones a realizar
    rate_inicial: el ratio incial de aprendizaje
    pesos_iniciales: lista de pesos iniciales por si se desea retomar el entrenamiento
    rate_decay: indica si el ratio debe decaer con el tiempo o mantenerse igual.
    estocastico: indica si se usa el metodo estocastico si es True o el metodo batch si es false
    summary: variable con la clase para almacenar los datos de las graficas
'''
def entrena(conjunto, resultados, clases, n_epochs, rate_inicial, pesos_iniciales, rate_decay, estocastico, summary=None):
    pesos = None
    rate = rate_inicial
    if pesos_iniciales == None:
        pesos = clasificador.genera_pesos(len(conjunto[0]))
    else:
        pesos = pesos_iniciales
    epoch = 0
    n_errors = 1
    while epoch < n_epochs and n_errors != 0:
        n_errors = 0
        for index in range(0, len(conjunto)):
            prediccion = clasificador.calcular_prediccion(conjunto[index], pesos, clases)
            if prediccion != resultados[index]:
                n_errors += 1
            if estocastico:
                pesos = ajusta_pesos_estocastico(conjunto[index], pesos, clasificador.busca_resultado(resultados[index], clases), rate)
        if not estocastico:
            pesos = ajusta_pesos_batch(conjunto, pesos, resultados, clases, rate)
        if rate_decay:
            rate = clasificador.decaer_ratio(rate_inicial, epoch)
        if summary != None:
            summary.add_epoch(n_errors/len(conjunto))
            summary.add_magnitud(clasificador.error_cuadratico(conjunto, resultados, pesos, clases))
        epoch += 1
    return pesos

'''
Metodo que implementa el ajuste de pesos estocastico para la regresion logistica
    conjunto: conjunto con los atributos de un elemento del conjunto de ejemplo
    pesos: lista de los pesos que vamos a ajustar
    esperado: resultado esperado para este elemento
    rate: ratio de aprendizaje
'''
def ajusta_pesos_estocastico(conjunto, pesos, esperado, rate):
    coef = []
    error = clasificador.sigma(clasificador.calcular_producto_escalar(pesos, conjunto))
    coef.append(pesos[0] + rate * (esperado - error) * error * (1 - error))
    for i in range(0, len(conjunto)):
        coef.append(pesos[i + 1] + rate * conjunto[i] * (esperado - error) * error * (1 - error))
    return coef

'''
Metodo que implementa el auste de pesos batch para la regresion logistica
    conjunto: conjunto de elementos del ejemplo
    pesos: lista de los pesos que vamos a ajustar
    resultados: lista de resultados esperado para este conjunto
    clases: posibles clases que pueden tomar los resultados
    rate: ratio de aprendizaje
'''
def ajusta_pesos_batch(conjunto, pesos, resultados, clases, rate):
    coef = copy.copy(pesos)
    sumatorio_cero = 0
    for j in range(0, len(conjunto)):
        objetivo = clasificador.busca_resultado(resultados[j], clases)
        sigma_cero = clasificador.sigma(clasificador.calcular_producto_escalar(pesos, conjunto[j]))
        sumatorio_cero += (objetivo - sigma_cero) * sigma_cero * (1 - sigma_cero)
    coef[0] = coef[0] + rate * sumatorio_cero
    for i in range(1, len(pesos)):
        error_global = 0
        for j in range(0, len(conjunto)):
            objetivo = clasificador.busca_resultado(resultados[j], clases)
            sigma = clasificador.sigma(clasificador.calcular_producto_escalar(pesos, conjunto[j]))
            error = (objetivo - sigma) * sigma * (1 - sigma) 
            error_global += error * conjunto[j][i - 1]
        coef[i] = coef[i] + (rate  * error_global)
    return coef
