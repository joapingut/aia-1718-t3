# -*- coding: utf-8 -*-

'''
Función que porcesa un archivo con los digitos para crear una lista con ellos.
    - archivo es la ruta del archivo
    - limit es el numero maximo de digitos a mirar.
'''
def procesarDigitosEscritos(archivo, limit):
    res = list()
    digit = list()
    file = open(archivo,"r")
    counter = 0
    for line in file:
        digit = digit + procesarDatos(line)
        if len(digit) == 784:
            res.append(digit)
            digit = list()
            counter += 1
            if counter >= limit:
                break
    return res

'''
Función que porcesa un caracter para devolver el valor.
Si el caracter es un espacio su valor es 0.
Si el caracter es un + su valor es 1
Si el caracter es # su valor es 1
'''
def procesarDatos(linea):
    res = list()
    for char in linea:
        if char == ' ':
            res.append(0)
        elif char == '+':
            res.append(1)
        elif char == '#':
            res.append(1)
    return res

'''
Función que procesa el archivo con los resultados de los digitos
para crear una lista con ellos.
'''
def procesarDigitos(archivo, limit):
    res = list()
    file = open(archivo,"r")
    counter = 0
    for line in file:
        res.append(line.split("\n")[0])
        counter += 1
        if counter >= limit:
            break
    return res


'''
Funcion que genera todos los archivos y los deposita en la ruta de destino.
La ruta debe tener este formato carpeta1/prefijo
los archivos estaran en la carpeta1 y todos empezaran por el prefijo indicado.
    - limit: limite para el conjunto de entrenamiento
    - limit_v: limite para el conjunto de validacion
    - limit_t: limite para el conjunto de test
EJ: crearArchivo("datos/digidata_p/digidata")
'''
def crearArchivo(dest, limit=100, limit_v=50, limit_t=10):
    
    file = open(dest+"_entr.py",'w', encoding='utf-8')
    
    file.write("digitdata_clases=['0','1','2','3','4','5','6','7','8','9']\n\n")
    file.write("digitdata_entr="+str(procesarDigitosEscritos("../datos/digitdata/trainingimages", limit))+"\n\n")
    file.write("digitdata_entr_clas="+str(procesarDigitos("../datos/digitdata/traininglabels", limit))+"\n\n")
    
    file.close()
    
    file = open(dest+"_valid.py",'w', encoding='utf-8')
    
    file.write("digitdata_clases=['0','1','2','3','4','5','6','7','8','9']\n\n")
    file.write("digitdata_valid="+str(procesarDigitosEscritos("../datos/digitdata/validationimages", limit_v))+"\n\n")
    file.write("digitdata_valid_clas="+str(procesarDigitos("../datos/digitdata/validationlabels", limit_v))+"\n\n")
    
    file.close()
    
    file = open(dest+"_test.py",'w', encoding='utf-8')
    
    file.write("digitdata_clases=['0','1','2','3','4','5','6','7','8','9']\n\n")
    file.write("digitdata_test="+str(procesarDigitosEscritos("../datos/digitdata/testimages", limit_t))+"\n\n")
    file.write("digitdata_test_clas="+str(procesarDigitos("../datos/digitdata/testlabels", limit_t))+"\n\n")
    
    file.close()
