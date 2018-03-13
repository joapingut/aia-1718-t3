# -*- coding: utf-8 -*-

__author__ = 'Joaquin'

import procesamiento as process

process.crearArchivo("datos/digidata_p/digidata", limit=500, limit_v=200, limit_t=200)

import clasificadores.perceptron as per
import clasificadores.maximizarVerosimilitud as maximizar
import clasificadores.regresionLogistica as regresion
import clasificadores.generadorConjuntos as Gen
import clasificadores.clasificador as Clasificador
import clasificadores.onevsrest as onevsrest
import datos.votos as Votos
import datos.digidata_p.digidata_entr as digi_entr
import datos.digidata_p.digidata_valid as digi_valid
import procesamiento as process


print("Crea ")
one_vs_rest = onevsrest.One_vs_Rest(digi_entr.digitdata_clases, "perceptron", estocastico=True, norm=False)
print("entrena")
one_vs_rest.entrena(digi_entr.digitdata_entr, digi_entr.digitdata_entr_clas, 1, rate=0.1, rate_decay=True)
print("valida")
print("One_vs_rest: ", one_vs_rest.evalua(digi_valid.digitdata_valid, digi_valid.digitdata_valid_clas))
print("One_vs_rest, esperado: ", digi_valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(digi_valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", digi_valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(digi_valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", digi_valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(digi_valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", digi_valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(digi_valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", digi_valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(digi_valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", digi_valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(digi_valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", digi_valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(digi_valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", digi_valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(digi_valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", digi_valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(digi_valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", digi_valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(digi_valid.digitdata_valid[7]))

normalizar = True

perceptron = per.Perceptron(Votos.votos_clases, norm=normalizar)

perceptron.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=True)
perceptron.summary.imprime_error()

print("Preceptron estocastico: ",perceptron.imprime())
print("Preceptron estocastico: ",perceptron.evalua(Votos.votos_valid, Votos.votos_valid_clas))


max_esto = maximizar.Maximizar(Votos.votos_clases, norm=normalizar, estocastico=True)
max_esto.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=True)
max_esto.summary.imprime_error()
max_esto.summary.imprime_magnitud()

print("Maximizar estocastico: ", max_esto.imprime())
print("Maximizar estocastico: ", max_esto.evalua(Votos.votos_valid, Votos.votos_valid_clas))

max_batch = maximizar.Maximizar(Votos.votos_clases, norm=normalizar, estocastico=False)
max_batch.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=False)
max_batch.summary.imprime_error()
max_batch.summary.imprime_magnitud()

print("Maximizar batch: ", max_batch.imprime())
print("Maximizar batch: ", max_batch.evalua(Votos.votos_valid, Votos.votos_valid_clas))

regre_esto = regresion.Regresion(Votos.votos_clases, norm=normalizar, estocastico=True)
regre_esto.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=True)
regre_esto.summary.imprime_error()
regre_esto.summary.imprime_magnitud()

print("Regresion estocastico: ", regre_esto.imprime())
print("Regresion estocastico: ", regre_esto.evalua(Votos.votos_valid, Votos.votos_valid_clas))

regre_batch = regresion.Regresion(Votos.votos_clases, norm=normalizar, estocastico=False)
regre_batch.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=True)
regre_batch.summary.imprime_error()
regre_batch.summary.imprime_magnitud()

print("Regresion batch: ", regre_batch.imprime())
print("Regresion batch: ", regre_batch.evalua(Votos.votos_valid, Votos.votos_valid_clas))


print("Otras pruebas")
print(Clasificador.genera_pesos(5))
# Pesos mejores para votos, probabilidad con validacion de 0.9710144927536232
mejor_votos = [0.03724722117863166, 0.07855852078341741, -0.04431696109736549, 0.016418011251744247, -0.23789171693988465, -0.042607598132008206, -0.014111082392901153, 0.021817553749997387, 0.04786968628491195, 0.0650495427619262, -0.049239096885685285, -0.015182104323912426, -0.07350127226732184, -0.03564717793892669, -0.00531360899617539, 0.03314275012576218, -0.021657938490079687]

perceptron.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000,pesos_iniciales=mejor_votos, rate_decay=False)
print(Votos.votos_test[0], Votos.votos_test_clas[0])
print(perceptron.clasifica(Votos.votos_test[0]))
print(perceptron.evalua(Votos.votos_valid, Votos.votos_valid_clas))

print(Gen.generar_conjunto_aleatorio(1, 3, 5))
print(Gen.generar_conjunto_aleatorio(1, 3, 5, clases=('Nay', 'Yay')))