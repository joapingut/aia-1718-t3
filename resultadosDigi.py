# -*- coding: utf-8 -*-


import clasificadores.onevsrest as onevsrest
import datos.digidata_p.digidata_valid as Valid
import datos.digidata_p.digidata_entr as Entr
import datos.digidata_p.digidata_test as Prue



'''
Pruebas para el conjunto de digidata

Los resultados de ejecuciones anteriores se encuentran en la carpeta pruebas/digidata/
y en el archivo excel está la puntuación de las diapoitivas.

Cuando salga el recuadro de una grafica el codigo no continuará hasta que se cierre.
'''

epochEstocastico = 50
epochBatch = 2

print("Sin normalización")

print("Creamos el clasificador One VS Rest, versión regresión logistica estocastica sin decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "regresion", estocastico=True, norm=False)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochEstocastico, rate=0.1, rate_decay=False)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

print("Creamos el clasificador One VS Rest, versión regresión logistica estocastica con decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "regresion", estocastico=True, norm=False)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochEstocastico, rate=0.1, rate_decay=True)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

print("Creamos el clasificador One VS Rest, versión regresión logistica batch sin decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "regresion", estocastico=False, norm=False)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochBatch, rate=0.1, rate_decay=False)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

print("Creamos el clasificador One VS Rest, versión regresión logistica batch con decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "regresion", estocastico=False, norm=False)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochBatch, rate=0.1, rate_decay=True)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))


print("Creamos el clasificador One VS Rest, versión maximizar verosimilitud estocastica sin decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "verosimilitud", estocastico=True, norm=False)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochEstocastico, rate=0.1, rate_decay=False)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

print("Creamos el clasificador One VS Rest, versión maximizar verosimilitud estocastica con decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "verosimilitud", estocastico=True, norm=False)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochEstocastico, rate=0.1, rate_decay=True)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

print("Creamos el clasificador One VS Rest, versión maximizar verosimilitud batch sin decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "verosimilitud", estocastico=False, norm=False)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochBatch, rate=0.1, rate_decay=False)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

print("Creamos el clasificador One VS Rest, versión maximizar verosimilitud batch con decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "verosimilitud", estocastico=False, norm=False)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochBatch, rate=0.1, rate_decay=True)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))



print("Con normalización")

print("Creamos el clasificador One VS Rest, versión regresión logistica estocastica sin decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "regresion", estocastico=True, norm=True)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochEstocastico, rate=0.1, rate_decay=False)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

print("Creamos el clasificador One VS Rest, versión regresión logistica estocastica con decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "regresion", estocastico=True, norm=True)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochEstocastico, rate=0.1, rate_decay=True)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

print("Creamos el clasificador One VS Rest, versión regresión logistica batch sin decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "regresion", estocastico=False, norm=True)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochBatch, rate=0.1, rate_decay=False)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

print("Creamos el clasificador One VS Rest, versión regresión logistica batch con decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "regresion", estocastico=False, norm=True)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochBatch, rate=0.1, rate_decay=True)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

'''

print("Creamos el clasificador One VS Rest, versión maximizar verosimilitud estocastica sin decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "verosimilitud", estocastico=True, norm=True)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochEstocastico, rate=0.1, rate_decay=False)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

print("Creamos el clasificador One VS Rest, versión maximizar verosimilitud estocastica con decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "verosimilitud", estocastico=True, norm=True)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochEstocastico, rate=0.1, rate_decay=True)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

print("Creamos el clasificador One VS Rest, versión maximizar verosimilitud batch sin decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "verosimilitud", estocastico=False, norm=True)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochBatch, rate=0.1, rate_decay=False)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

print("Creamos el clasificador One VS Rest, versión maximizar verosimilitud batch con decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "verosimilitud", estocastico=False, norm=True)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochBatch, rate=0.1, rate_decay=True)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

'''

print("Sin normalización")

print("Creamos el clasificador One VS Rest, versión perceptron estocastica sin decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "perceptron", estocastico=True, norm=False)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochBatch, rate=0.1, rate_decay=False)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))


print("Creamos el clasificador One VS Rest, versión perceptron estocastica con decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "perceptron", estocastico=True, norm=False)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochEstocastico, rate=0.1, rate_decay=True)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))

print("Con normalización")

print("Creamos el clasificador One VS Rest, versión perceptron estocastica sin decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "perceptron", estocastico=True, norm=True)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochBatch, rate=0.1, rate_decay=False)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))


print("Creamos el clasificador One VS Rest, versión perceptron estocastica con decaimiento")

one_vs_rest = onevsrest.One_vs_Rest(Entr.digitdata_clases, "perceptron", estocastico=True, norm=True)
one_vs_rest.entrena(Entr.digitdata_entr, Entr.digitdata_entr_clas, epochEstocastico, rate=0.1, rate_decay=True)

print("Validación: ", one_vs_rest.evalua(Valid.digitdata_valid, Valid.digitdata_valid_clas))
print("Pruebas: ", one_vs_rest.evalua(Prue.digitdata_test, Prue.digitdata_test_clas))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[3], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[3]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[2], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[2]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[1], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[1]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[18], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[18]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[4], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[4]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[8], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[8]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[11], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[11]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[0], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[0]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[61], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[61]))
print("One_vs_rest, esperado: ", Valid.digitdata_valid_clas[7], " obtenido: ", one_vs_rest.clasifica(Valid.digitdata_valid[7]))
