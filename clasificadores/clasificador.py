# -*- coding: utf-8 -*-

import random, copy, math, numpy
import matplotlib.pyplot as plt

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


'''
Clase para guardar y pintar las graficas con el numero de errores y la magnitud
que se desea mejorar.
'''
class Summary:

    def __init__(self, name):
        self.errores = []
        self.magnitudes = []
        self.name = name

    def add_epoch(self, error):
        self.errores.append(error)

    def add_magnitud(self, magnitud):
        self.magnitudes.append(magnitud)

    def imprime_error(self):
        plt.plot(range(0,len(self.errores)), self.errores,marker='o')
        plt.title(self.name)
        plt.xlabel('Epochs')
        plt.ylabel('Porcentaje de errores')
        plt.show()

    def imprime_magnitud(self):
        plt.plot(range(0,len(self.magnitudes)), self.magnitudes,marker='o')
        plt.title(self.name)
        plt.xlabel('Epochs')
        plt.ylabel('Magnitud del error')
        plt.show()


'''
Metodo para dado un conjunto de atributos, los pesos, las clases y el metodo para la evalucion
devuelve la prediccion para dicho conjnto de atributos.
Si clases es None se devulve el valor numerico de la prediccion que para sigam es un numero
entre 0 y 1 y para el umbral es 0 o 1
'''
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

'''
Funcion para decaer el ratio de aprendizaje.
'''
def decaer_ratio(rate, epoch):
    #return rate + 2/(((epoch + 1) ** 2) **(1.0/3.0))
    return rate + 2/(math.pow(epoch + 1, 2/3))


'''
Metodo auxiliar para ver el indice que le corresponde a la lista de clases a la
clase que estoy buscando en concreto.
'''
def busca_resultado(busco, clases):
    for index in range(0, len(clases)):
        if clases[index] == busco:
            return index
    return None

'''
Metodo que calcula el producto escalar entre la lista de pesos y el conjunto
de atributos.
'''
def calcular_producto_escalar(pesos, atributos):
    coef = pesos[0]
    for i in range(0, len(atributos)):
        coef += pesos[i + 1] * atributos[i]
    return coef

'''
Metodo que implementa la funcion del umbral.
Para numeros menores de 0 devuelve 0 y para mayores o igual 1
'''
def umbral(num):
    if num >= 0:
        return 1
    return 0

'''
Metodo que implementa la funcion sigma que devuelve valores entre
0 y 1 dependiendo del valor z pasado.
'''
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

'''
Metodo que dado un conjunto de ejemplos devuelve una tupla
con la media por columnas y la desviacion tipica por columnas
de todos los elementos del ejemplo.
'''
def extrae_normalizacion(conjunto):
    mean = []
    desviacion = []
    num_attr = len(conjunto[0])
    nun_arra = numpy.asarray(conjunto)
    for j in range(0, num_attr):
        mean.append(numpy.mean(nun_arra[:,j]))
        desviacion.append(numpy.std(nun_arra[:,j]))
    return (mean, desviacion)

'''
Metodo que dado una lista con ejemplos con sus atributos, la lista de las medias
por columnas y la de las desviaciones normaliza los atributos de todos los elementos
que hay en dicho ejemplo.
'''
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


'''
Metodo que dado un ejemplo con sus atributos, la lista de las medias
por columnas y la de las desviaciones normaliza los atributos de dicho
ejemplo.
'''
def normalizar_elemento(ejemplo, media, desviacion):
    norm_arra = []
    for j in range(0, len(ejemplo)):
        num = ejemplo[j]
        nnum = puntuacion_estandar(num,media[j],desviacion[j])
        norm_arra.append(nnum)
    return norm_arra


'''
Metodo que calcula la punciacion estandar, es decir, normaliza un numero
en funcion de la media y desviacion tipica
'''
def puntuacion_estandar(num, media, desvia):
    return (num - media) / desvia


'''
Metodo que calcular el error cuadratico de una lista con elementos con sus
atributos, los resultados esperados para cada elemento, los pesos y las clases
que pueden tener los resultados.
'''
def error_cuadratico(ejemplos, resultados, pesos, clases):
    sumatorio = 0
    for i in range(0, len(ejemplos)):
        y = busca_resultado(resultados[i], clases)
        sigm = sigma(calcular_producto_escalar(pesos, ejemplos[i]))
        sumatorio += math.pow((y - sigm),2)
    return sumatorio

'''
Metodo que calcula la log-verosimilitud de una lista de elementos con sus atributos
y la lista de pesos.
'''
def error_verosimilitud(ejemplos, pesos):
    sumatorio_positivo = 0
    sumatorio_negativo = 0
    for i in range(0, len(ejemplos)):
        producto = calcular_producto_escalar(pesos, ejemplos[i])
        sumatorio_negativo += math.log10(1 + math.exp(-producto))
        sumatorio_positivo += math.log10(1 + math.exp(producto))
    return - sumatorio_negativo - sumatorio_positivo