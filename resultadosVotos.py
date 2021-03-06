# -*- coding: utf-8 -*-


import clasificadores.perceptron as per
import clasificadores.maximizarVerosimilitud as maximizar
import clasificadores.regresionLogistica as regresion
import clasificadores.generadorConjuntos as Gen
import clasificadores.clasificador as Clasificador
import datos.votos as Votos



'''
Pruebas para el conjunto de votos.py

Los reusltados de ejecuciones anteriores se encuentran en la carpeta pruebas/votos/
y en el archivo excel está la puntuación de las diapoitivas.

Cuando salga el recuadro de una grafica el codigo no continuará hasta que se cierre.

Las pruebas se hacen en bloques separando por las variables normalizar y rate_decay.
Cada grupo ejecuta todos los algoritmos que hemos usado en clase.

Como se explica en las diapositivas, la versión batch de la maximización de la verosimilitud
da errores a la hora de pintar las graficas porque los valores se vuelve muy grandes y no se
puede calcular correctamente la log-verosimilitud.
'''


normalizar = False
rate_decay = False

print("Pruebas sin normalizacion y sin decaimiento del ratio")

perceptron = per.Perceptron(Votos.votos_clases, norm=normalizar)

perceptron.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
perceptron.summary.imprime_error()

print("Preceptron estocastico: ",perceptron.imprime())
print("Preceptron estocastico: ",perceptron.evalua(Votos.votos_valid, Votos.votos_valid_clas))


max_esto = maximizar.Maximizar(Votos.votos_clases, norm=normalizar, estocastico=True)
max_esto.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
max_esto.summary.imprime_error()
max_esto.summary.imprime_magnitud()

print("Maximizar estocastico: ", max_esto.imprime())
print("Maximizar estocastico: ", max_esto.evalua(Votos.votos_valid, Votos.votos_valid_clas))

max_batch = maximizar.Maximizar(Votos.votos_clases, norm=normalizar, estocastico=False)
max_batch.summary = None
max_batch.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
#max_batch.summary.imprime_error()
#max_batch.summary.imprime_magnitud()

print("Maximizar batch: ", max_batch.imprime())
print("Maximizar batch: ", max_batch.evalua(Votos.votos_valid, Votos.votos_valid_clas))

regre_esto = regresion.Regresion(Votos.votos_clases, norm=normalizar, estocastico=True)
regre_esto.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
regre_esto.summary.imprime_error()
regre_esto.summary.imprime_magnitud()

print("Regresion estocastico: ", regre_esto.imprime())
print("Regresion estocastico: ", regre_esto.evalua(Votos.votos_valid, Votos.votos_valid_clas))

regre_batch = regresion.Regresion(Votos.votos_clases, norm=normalizar, estocastico=False)
regre_batch.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
regre_batch.summary.imprime_error()
regre_batch.summary.imprime_magnitud()

print("Regresion batch: ", regre_batch.imprime())
print("Regresion batch: ", regre_batch.evalua(Votos.votos_valid, Votos.votos_valid_clas))


normalizar = False
rate_decay = True

print("Pruebas sin normalizacion y con decaimiento del ratio")

perceptron = per.Perceptron(Votos.votos_clases, norm=normalizar)

perceptron.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
perceptron.summary.imprime_error()

print("Preceptron estocastico: ",perceptron.imprime())
print("Preceptron estocastico: ",perceptron.evalua(Votos.votos_valid, Votos.votos_valid_clas))


max_esto = maximizar.Maximizar(Votos.votos_clases, norm=normalizar, estocastico=True)
max_esto.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
max_esto.summary.imprime_error()
max_esto.summary.imprime_magnitud()

print("Maximizar estocastico: ", max_esto.imprime())
print("Maximizar estocastico: ", max_esto.evalua(Votos.votos_valid, Votos.votos_valid_clas))

max_batch = maximizar.Maximizar(Votos.votos_clases, norm=normalizar, estocastico=False)
max_batch.summary = None
max_batch.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
#max_batch.summary.imprime_error()
#max_batch.summary.imprime_magnitud()

print("Maximizar batch: ", max_batch.imprime())
print("Maximizar batch: ", max_batch.evalua(Votos.votos_valid, Votos.votos_valid_clas))

regre_esto = regresion.Regresion(Votos.votos_clases, norm=normalizar, estocastico=True)
regre_esto.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
regre_esto.summary.imprime_error()
regre_esto.summary.imprime_magnitud()

print("Regresion estocastico: ", regre_esto.imprime())
print("Regresion estocastico: ", regre_esto.evalua(Votos.votos_valid, Votos.votos_valid_clas))

regre_batch = regresion.Regresion(Votos.votos_clases, norm=normalizar, estocastico=False)
regre_batch.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
regre_batch.summary.imprime_error()
regre_batch.summary.imprime_magnitud()

print("Regresion batch: ", regre_batch.imprime())
print("Regresion batch: ", regre_batch.evalua(Votos.votos_valid, Votos.votos_valid_clas))

normalizar = True
rate_decay = False

print("Pruebas con normalizacion y sin decaimiento del ratio")

perceptron = per.Perceptron(Votos.votos_clases, norm=normalizar)

perceptron.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
perceptron.summary.imprime_error()

print("Preceptron estocastico: ",perceptron.imprime())
print("Preceptron estocastico: ",perceptron.evalua(Votos.votos_valid, Votos.votos_valid_clas))


max_esto = maximizar.Maximizar(Votos.votos_clases, norm=normalizar, estocastico=True)
max_esto.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
max_esto.summary.imprime_error()
max_esto.summary.imprime_magnitud()

print("Maximizar estocastico: ", max_esto.imprime())
print("Maximizar estocastico: ", max_esto.evalua(Votos.votos_valid, Votos.votos_valid_clas))

max_batch = maximizar.Maximizar(Votos.votos_clases, norm=normalizar, estocastico=False)
max_batch.summary = None
max_batch.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
#max_batch.summary.imprime_error()
#max_batch.summary.imprime_magnitud()

print("Maximizar batch: ", max_batch.imprime())
print("Maximizar batch: ", max_batch.evalua(Votos.votos_valid, Votos.votos_valid_clas))

regre_esto = regresion.Regresion(Votos.votos_clases, norm=normalizar, estocastico=True)
regre_esto.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
regre_esto.summary.imprime_error()
regre_esto.summary.imprime_magnitud()

print("Regresion estocastico: ", regre_esto.imprime())
print("Regresion estocastico: ", regre_esto.evalua(Votos.votos_valid, Votos.votos_valid_clas))

regre_batch = regresion.Regresion(Votos.votos_clases, norm=normalizar, estocastico=False)
regre_batch.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
regre_batch.summary.imprime_error()
regre_batch.summary.imprime_magnitud()

print("Regresion batch: ", regre_batch.imprime())
print("Regresion batch: ", regre_batch.evalua(Votos.votos_valid, Votos.votos_valid_clas))

normalizar = True
rate_decay = True

print("Pruebas con normalizacion y con decaimiento del ratio")

perceptron = per.Perceptron(Votos.votos_clases, norm=normalizar)

perceptron.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
perceptron.summary.imprime_error()

print("Preceptron estocastico: ",perceptron.imprime())
print("Preceptron estocastico: ",perceptron.evalua(Votos.votos_valid, Votos.votos_valid_clas))


max_esto = maximizar.Maximizar(Votos.votos_clases, norm=normalizar, estocastico=True)
max_esto.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
max_esto.summary.imprime_error()
max_esto.summary.imprime_magnitud()

print("Maximizar estocastico: ", max_esto.imprime())
print("Maximizar estocastico: ", max_esto.evalua(Votos.votos_valid, Votos.votos_valid_clas))

max_batch = maximizar.Maximizar(Votos.votos_clases, norm=normalizar, estocastico=False)
max_batch.summary = None
max_batch.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
#max_batch.summary.imprime_error()
#max_batch.summary.imprime_magnitud()

print("Maximizar batch: ", max_batch.imprime())
print("Maximizar batch: ", max_batch.evalua(Votos.votos_valid, Votos.votos_valid_clas))

regre_esto = regresion.Regresion(Votos.votos_clases, norm=normalizar, estocastico=True)
regre_esto.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
regre_esto.summary.imprime_error()
regre_esto.summary.imprime_magnitud()

print("Regresion estocastico: ", regre_esto.imprime())
print("Regresion estocastico: ", regre_esto.evalua(Votos.votos_valid, Votos.votos_valid_clas))

regre_batch = regresion.Regresion(Votos.votos_clases, norm=normalizar, estocastico=False)
regre_batch.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=rate_decay)
regre_batch.summary.imprime_error()
regre_batch.summary.imprime_magnitud()

print("Regresion batch: ", regre_batch.imprime())
print("Regresion batch: ", regre_batch.evalua(Votos.votos_valid, Votos.votos_valid_clas))