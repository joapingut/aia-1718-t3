# -*- coding: utf-8 -*-

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

def procesarDatos(linea):
    res = list()
    for char in linea:
        if char == ' ':
            res.append(0)
        elif char == '+':
            res.append(1)
        elif char == '#':
            res.append(-1)
    return res

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

def crearArchivo(dest):
    
    file = open(dest+"_entr.py",'w', encoding='utf-8')
    
    file.write("digitdata_clases=['0','1','2','3','4','5','6','7','8','9']\n\n")
    file.write("digitdata_entr="+str(procesarDigitosEscritos("datos/digitdata/trainingimages", 200))+"\n\n")
    file.write("digitdata_entr_clas="+str(procesarDigitos("datos/digitdata/traininglabels", 200))+"\n\n")
    
    file.close()
    
    file = open(dest+"_valid.py",'w', encoding='utf-8')
    
    file.write("digitdata_clases=['0','1','2','3','4','5','6','7','8','9']\n\n")
    file.write("digitdata_valid="+str(procesarDigitosEscritos("datos/digitdata/validationimages", 50))+"\n\n")
    file.write("digitdata_valid_clas="+str(procesarDigitos("datos/digitdata/validationlabels", 50))+"\n\n")
    
    file.close()
    
    file = open(dest+"_test.py",'w', encoding='utf-8')
    
    file.write("digitdata_clases=['0','1','2','3','4','5','6','7','8','9']\n\n")
    file.write("digitdata_test="+str(procesarDigitosEscritos("datos/digitdata/testimages", 50))+"\n\n")
    file.write("digitdata_test_clas="+str(procesarDigitos("datos/digitdata/testlabels", 50))+"\n\n")
    
    file.close()