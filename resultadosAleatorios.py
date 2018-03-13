# -*- coding: utf-8 -*-


import clasificadores.perceptron as per
import clasificadores.maximizarVerosimilitud as maximizar
import clasificadores.regresionLogistica as regresion
import clasificadores.generadorConjuntos as generador


'''
Pruebas para el conjunto de datos generados aleatoriamente

Los reusltados de ejecuciones anteriores se encuentran en la carpeta pruebas/aleatorios/
y en el archivo excel está la puntuación de las diapositivas.

Cuando salga el recuadro de una grafica el codigo no continuará hasta que se cierre.

Las pruebas se hacen en bloques separando por nu y rate_decay.
Cada grupo ejecuta todos los algoritmos que hemos usado en clase.

Como se explica en las diapositivas, la versión batch de la maximización de la verosimilitud
da errores a la hora de pintar las graficas porque los valores se vuelve muy grandes y no se
puede calcular correctamente la log-verosimilitud.
'''

'''
Generamos conjuntos de pruebas, validacion y test aleatorios usando la misma lista de pesos
para que todos sean sepaables de la misma forma.
'''
clases = ['Clase 1', 'Clase 2']
conjunto = generador.generar_conjunto_aleatorio(1, 20, 1000, separable=True, clases=clases)
conjunto_pesos_ideal = conjunto[0]
conjunto_entrenamiento = conjunto[1]
conjunto_entrenamiento_class = conjunto[2]
conjunto = generador.generar_conjunto_aleatorio(1, 20, 200, separable=True, clases=clases, pesos=conjunto_pesos_ideal)
conjunto_validacion = conjunto[1]
conjunto_validacion_class = conjunto[2]
conjunto = generador.generar_conjunto_aleatorio(1, 20, 200, separable=True, clases=clases, pesos=conjunto_pesos_ideal)
conjunto_test = conjunto[1]
conjunto_test_class = conjunto[2]

print("Pesos ideales: ", conjunto_pesos_ideal)

rate_decay = False
normalizar = False
rate = 0.1

print("Sin decaimiento del ratio")

perceptron = per.Perceptron(clases, norm=normalizar)

perceptron.entrena(conjunto_entrenamiento, conjunto_entrenamiento_class, 1000, rate=rate, rate_decay=rate_decay)
perceptron.summary.imprime_error()

print("Preceptron estocastico: ",perceptron.imprime())
print("Preceptron estocastico validacion: ",perceptron.evalua(conjunto_validacion, conjunto_validacion_class))
print("Preceptron estocastico test: ",perceptron.evalua(conjunto_test, conjunto_test_class))


max_esto = maximizar.Maximizar(clases, norm=normalizar, estocastico=True)
max_esto.entrena(conjunto_entrenamiento, conjunto_entrenamiento_class, 1000, rate_decay=rate_decay)
max_esto.summary.imprime_error()
max_esto.summary.imprime_magnitud()

print("Maximizar estocastico: ", max_esto.imprime())
print("Maximizar estocastico: ", max_esto.evalua(conjunto_validacion, conjunto_validacion_class))
print("Maximizar estocastico test: ", max_esto.evalua(conjunto_test, conjunto_test_class))

max_batch = maximizar.Maximizar(clases, norm=normalizar, estocastico=False)
max_batch.summary = None
max_batch.entrena(conjunto_entrenamiento, conjunto_entrenamiento_class, 1000, rate_decay=rate_decay)
#max_batch.summary.imprime_error()
#max_batch.summary.imprime_magnitud()

print("Maximizar batch: ", max_batch.imprime())
print("Maximizar batch: ", max_batch.evalua(conjunto_validacion, conjunto_validacion_class))
print("Maximizar estocastico test: ",max_batch.evalua(conjunto_test, conjunto_test_class))

regre_esto = regresion.Regresion(clases, norm=normalizar, estocastico=True)
regre_esto.entrena(conjunto_entrenamiento, conjunto_entrenamiento_class, 1000, rate_decay=rate_decay)
regre_esto.summary.imprime_error()
regre_esto.summary.imprime_magnitud()

print("Regresion estocastico: ", regre_esto.imprime())
print("Regresion estocastico: ", regre_esto.evalua(conjunto_validacion, conjunto_validacion_class))
print("Regresion estocastico test: ",regre_esto.evalua(conjunto_test, conjunto_test_class))

regre_batch = regresion.Regresion(clases, norm=normalizar, estocastico=False)
regre_batch.entrena(conjunto_entrenamiento, conjunto_entrenamiento_class, 1000, rate_decay=rate_decay)
regre_batch.summary.imprime_error()
regre_batch.summary.imprime_magnitud()

print("Regresion batch: ", regre_batch.imprime())
print("Regresion batch: ", regre_batch.evalua(conjunto_validacion, conjunto_validacion_class))
print("Regresion estocastico test: ",regre_batch.evalua(conjunto_test, conjunto_test_class))


normalizar = False
rate_decay = True

print("Pruebas con decaimiento del ratio")

perceptron = per.Perceptron(clases, norm=normalizar)

perceptron.entrena(conjunto_entrenamiento, conjunto_entrenamiento_class, 1000, rate_decay=rate_decay)
perceptron.summary.imprime_error()

print("Preceptron estocastico: ",perceptron.imprime())
print("Preceptron estocastico: ",perceptron.evalua(conjunto_validacion, conjunto_validacion_class))
print("Preceptron estocastico test: ",perceptron.evalua(conjunto_test, conjunto_test_class))


max_esto = maximizar.Maximizar(clases, norm=normalizar, estocastico=True)
max_esto.entrena(conjunto_entrenamiento, conjunto_entrenamiento_class, 1000, rate_decay=rate_decay)
max_esto.summary.imprime_error()
max_esto.summary.imprime_magnitud()

print("Maximizar estocastico: ", max_esto.imprime())
print("Maximizar estocastico: ", max_esto.evalua(conjunto_validacion, conjunto_validacion_class))
print("Maximizar estocastico test: ",max_esto.evalua(conjunto_test, conjunto_test_class))

max_batch = maximizar.Maximizar(clases, norm=normalizar, estocastico=False)
max_batch.summary = None
max_batch.entrena(conjunto_entrenamiento, conjunto_entrenamiento_class, 1000, rate_decay=rate_decay)
#max_batch.summary.imprime_error()
#max_batch.summary.imprime_magnitud()

print("Maximizar batch: ", max_batch.imprime())
print("Maximizar batch: ", max_batch.evalua(conjunto_validacion, conjunto_validacion_class))
print("Maximizar estocastico test: ",max_batch.evalua(conjunto_test, conjunto_test_class))

regre_esto = regresion.Regresion(clases, norm=normalizar, estocastico=True)
regre_esto.entrena(conjunto_entrenamiento, conjunto_entrenamiento_class, 1000, rate_decay=rate_decay)
regre_esto.summary.imprime_error()
regre_esto.summary.imprime_magnitud()

print("Regresion estocastico: ", regre_esto.imprime())
print("Regresion estocastico: ", regre_esto.evalua(conjunto_validacion, conjunto_validacion_class))
print("Regresion estocastico test: ",regre_esto.evalua(conjunto_test, conjunto_test_class))

regre_batch = regresion.Regresion(clases, norm=normalizar, estocastico=False)
regre_batch.entrena(conjunto_entrenamiento, conjunto_entrenamiento_class, 1000, rate_decay=rate_decay)
regre_batch.summary.imprime_error()
regre_batch.summary.imprime_magnitud()

print("Regresion batch: ", regre_batch.imprime())
print("Regresion batch: ", regre_batch.evalua(conjunto_validacion, conjunto_validacion_class))
print("Regresion estocastico test: ",regre_batch.evalua(conjunto_test, conjunto_test_class))