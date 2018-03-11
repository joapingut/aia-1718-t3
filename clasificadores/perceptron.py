# -*- coding: utf-8 -*-

from clasificadores.clasificador import Clasificador
import clasificadores.clasificador as clasificador

'''
Clase que implementa un clasificador por medio del metodo del perceptron
'''
class Perceptron(Clasificador):

    '''
    Constructor, se le debe pasar las clases.
    Opcionalmente se puede indicar si se quiere normalizar los datos.
    '''
    def __init__(self, clases, norm=False):
        self.clases = clases
        self.normaliza = norm
        self.entrenado = False
        self.pesos = None
        self.media = None
        self.desviacion = None
        self.summary = clasificador.Summary("Perceptron")
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
        self.pesos = entrena(conjunto, clas_entr, self.clases, n_epochs, rate, pesos_iniciales, rate_decay, self.summary)
        self.entrenado = True
        None

    '''
    Metodo que dado un ejemplo indica la probabilidad de que tiene de pertenecer a una clase y otra.
    El perceptron solo devuelve valores que pueden ser 0 o 1 ya que se usa la funcion umbral siendo
    el 0 la clase en la posicion 0 de la lista del cosntructor y el 1 la clase del indice 1
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
    summary: variable con la clase para almacenar los datos de las graficas
'''
def entrena(conjunto, resultados, clases, n_epochs, rate_inicial, pesos_iniciales, rate_decay, summary=None):
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
            prediccion = clasificador.calcular_prediccion(conjunto[index], pesos, clases, is_sigma=False)
            if prediccion != resultados[index]:
                n_errors += 1
            pesos = ajusta_pesos(conjunto[index], pesos, clasificador.busca_resultado(resultados[index], clases), rate)
        if rate_decay:
            rate = clasificador.decaer_ratio(rate_inicial, epoch)
        if summary != None:
            summary.add_epoch(n_errors/len(conjunto))
        epoch += 1
    return pesos

'''
Metodo que implementa el ajuste de pesos estocastico para el preceptron
    conjunto: conjunto con los atributos de un elemento del conjunto de ejemplo
    pesos: lista de los pesos que vamos a ajustar
    esperado: resultado esperado para este elemento
    rate: ratio de aprendizaje
'''
def ajusta_pesos(conjunto, pesos, esperado, rate):
    coef = []
    error = clasificador.umbral(clasificador.calcular_producto_escalar(pesos, conjunto))
    coef.append(pesos[0] + rate * (esperado - error))
    for i in range(0, len(conjunto)):
        coef.append(pesos[i + 1] + rate * conjunto[i] * (esperado - error))
    return coef
