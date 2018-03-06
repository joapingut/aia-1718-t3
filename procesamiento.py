# -*- coding: utf-8 -*-

def procesarDigitosEscritos(archivo):
    res = list()
    digit = list()
    file = open(archivo,"r")
    for line in file:
        digit = digit + procesarDatos(line)
        if len(digit) == 784:
            res.append(digit)
            digit = list()
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

def procesarDigitos(archivo):
    res = list()
    file = open(archivo,"r")
    for line in file:
        res.append(line.split("\n")[0])
    return res

def crearArchivo(dest):
    
    file = open(dest+"_entr.py",'w', encoding='utf-8')
    
    file.write("digitdata_clases=['0','1','2','3','4','5','6','7','8','9']\n\n")
    file.write("digitdata_entr="+str(procesarDigitosEscritos("datos/digitdata/trainingimages"))+"\n\n")
    file.write("digitdata_entr_clas="+str(procesarDigitos("datos/digitdata/traininglabels"))+"\n\n")
    
    file.close()
    
    file = open(dest+"_valid.py",'w', encoding='utf-8')
    
    file.write("digitdata_clases=['0','1','2','3','4','5','6','7','8','9']\n\n")
    file.write("digitdata_valid="+str(procesarDigitosEscritos("datos/digitdata/validationimages"))+"\n\n")
    file.write("digitdata_valid_clas="+str(procesarDigitos("datos/digitdata/validationlabels"))+"\n\n")
    
    file.close()
    
    file = open(dest+"_test.py",'w', encoding='utf-8')
    
    file.write("digitdata_clases=['0','1','2','3','4','5','6','7','8','9']\n\n")
    file.write("digitdata_test="+str(procesarDigitosEscritos("datos/digitdata/testimages"))+"\n\n")
    file.write("digitdata_test_clas="+str(procesarDigitos("datos/digitdata/testlabels"))+"\n\n")
    
    file.close()