# -*- coding: utf-8 -*-

__author__ = 'Joaquin'


import clasificadores.perceptron as per
import clasificadores.maximizarVerosimilitud as maximizar
import clasificadores.generadorConjuntos as Gen
import clasificadores.clasificador as Clasificador
import datos.votos as Votos


print(Clasificador.genera_pesos(5))

perceptron = per.Perceptron(Votos.votos_clases)

perceptron.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=False)

print(perceptron.imprime())
print(perceptron.evalua(Votos.votos_valid, Votos.votos_valid_clas))

# Pesos mejores para votos, probabilidad con validacion de 0.9710144927536232
mejor_votos = [0.03724722117863166, 0.07855852078341741, -0.04431696109736549, 0.016418011251744247, -0.23789171693988465, -0.042607598132008206, -0.014111082392901153, 0.021817553749997387, 0.04786968628491195, 0.0650495427619262, -0.049239096885685285, -0.015182104323912426, -0.07350127226732184, -0.03564717793892669, -0.00531360899617539, 0.03314275012576218, -0.021657938490079687]

perceptron.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000,pesos_iniciales=mejor_votos, rate_decay=False)
print(Votos.votos_test[0], Votos.votos_test_clas[0])
print(perceptron.clasifica(Votos.votos_test[0]))
print(perceptron.evalua(Votos.votos_valid, Votos.votos_valid_clas))

print(Gen.generar_conjunto_aleatorio(1, 3, 5))
print(Gen.generar_conjunto_aleatorio(1, 3, 5, clases=('Nay', 'Yay')))

max_esto = maximizar.Maximizar(Votos.votos_clases, estocastico=True)
max_esto.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=False)

print("Maximizar estocastico: ", max_esto.imprime())
print("Maximizar estocastico: ", max_esto.evalua(Votos.votos_valid, Votos.votos_valid_clas))

max_batch = maximizar.Maximizar(Votos.votos_clases, estocastico=False)
max_batch.entrena(Votos.votos_entr, Votos.votos_entr_clas, 1000, rate_decay=False)

print("Maximizar batch: ", max_batch.imprime())
print("Maximizar batch: ", max_batch.evalua(Votos.votos_valid, Votos.votos_valid_clas))