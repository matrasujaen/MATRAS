'''
Created on 3/2/2016

@author: Grupo MATRAS. Universidad de Jaen

-Modificaciones:
    20160203    version inicial.
'''
import numpy as np
from matplotlib import pyplot as plt

#===============================================================================
# CONSTANTES
#===============================================================================
nullValue = np.nan

#===============================================================================
# FUNCIONES
#===============================================================================
'''
--Descripcion:
La funcion 'filterOutliers' marca los valores atipicos (outliers) de un set de datos
como valores nulos ('nullValue'). En el calculo de los outliers no se tienen en cuenta
los valores enmascarados.

La teoria para detectar valores atipicos se ha obtenido del siguiente enlace:
    http://es.wikipedia.org/wiki/Valor_at%C3%ADpico

--Parametros de entrada:
    - dataset      numpy array    set de datos a evaluar.
    - factor       float          (OPT) factor de discriminacion, donde un valor 1.5 es para
eliminar un valor atipico leve y un factor 3.0 sera para eliminar un valor atipico
extremo.
    - maskValue    float          (OPT) indentificador de valores a enmascarar para no
tenerlos en cuenta en el calculo de los outliers.
    - debug        boolean        (OPT) flag para mostrar resultados intermedios.

--Parametros de salida:
    - datasetResul    numpy array    set de datos evaluado con outliers marcados como
'nullValue'.
'''
class outlierFilter(object):
    def __init__(self, dataset, factor = 1.5, maskValue = nullValue, verbose = False):
        '''
        Filtro los valores de 'factor' a los valores admitidos: 1.5, 3.0.
        '''
        if factor != 1.5 and factor != 3.0:
            return
        '''
        Creo una copia de 'dataset'.
        '''
        self.dataset = dataset.copy()
#         datasetResul = np.array(dataset, dtype = 'float')
        '''
        Creo un array con la misma forma que el array de entrada 'dataset'
        pero de tipo bool.
        '''
        self.mask = np.array(dataset, dtype='bool')
        '''
        Igualo todos los valores del array 'datasetValid' a True
        indicando que, de partida, todos los valores que contiene
        son validos.
        '''
        self.mask[:] = True
        '''
        No se tienen en cuenta los valores enmascarados, por lo
        que indico en su puesto un valor False.
        '''
        self.mask[self.dataset == maskValue] = False
        '''
        Calculo el cuartil 1 unicamente teniendo en cuenta los
        valores validos.
        '''
        q1 = np.percentile(self.dataset[self.mask == True], 25)
        '''
        Calculo el cuartil 3 unicamente teniendo en cuenta los
        valores admitidos.
        '''
        q3 = np.percentile(self.dataset[self.mask == True], 75)
        '''
        Calculo el rango inter-cuartilico.
        '''
        iQR = q3 - q1
        '''
        Calculo el valor minimo que delimita el conjunto de valores validos.
        '''
        vMin = q1 - (factor * iQR)
        '''
        Calculo el valor maximo que delimita el conjunto de valores validos.
        '''
        vMax = q3 + (factor * iQR)
        '''
        Igualo los valores de 'datasetValid' que se corresponden a
        valores de 'dataset' que estan mas alla de los valores
        minimo y maximo permitidos a un valor 'False'.
        '''
        self.mask[(self.dataset < vMin) | (self.dataset > vMax)] = False
        '''
        Muestro los resultados tanto finales como intermedios.
        '''
        if verbose == True:
            plt.title('outliers filter\nfactor = '+str(factor))
            for contItem in range(len(self.dataset)):
                if self.mask[contItem] == False:
                    plt.plot(contItem, self.dataset[contItem], color = 'red', marker = 'o')
                else:
                    plt.plot(contItem, self.dataset[contItem], color = 'green', marker = 'o')
            plt.hlines(vMin, 0, len(self.dataset))
            plt.hlines(vMax, 0, len(self.dataset))
            plt.show()

'''
                    2                                 1                            3
  (                                                                     )
  |    (newRangeTop - newRangeBottom)                                   |
  | ( -------------------------------- ) * (inputMat - oldRangeBottom)  |  + newRangeBottom
  |    (oldRangeTop - oldRangeBottom)                                   |
  (                                                                     )
  
  Supongamos que:
      newRangeTop = 3
      newRangeBottom = 1
      oldRangeTop = 30
      oldRangeBottom = 1
  
  + La resta '1' situa los valores de inputMat entre [0, 29].
  + En '2', la division entre (30 -1) situa los valores entre [0, 1].
  + En '2', la multiplicacion situa los valores entre [0, (3-1)].
  + En '3', el conjunto se situa entre [1, 3].
'''
def changeRange(inputMat, oldRangeBottom, oldRangeTop, newRangeBottom, newRangeTop):
    '''
    Me aparece el siguiente mensaje de warning:
    'RuntimeWarning: divide by zero encountered in longdouble_scalars'.
    Si hay una division por 0 es debido a que 'np.float128(oldRangeTop - oldRangeBottom)'
    es 0, lo que significa que todos los valores de la matriz son el mismo.
    Esta situacion parece ser tambien la causante del mensaje de warning:
    'RuntimeWarning: invalid value encountered in multiply'.
    Decido controlar esta situacion en la llamada a esta funcion.
    '''
    return (((newRangeTop - newRangeBottom)/np.float128(oldRangeTop - oldRangeBottom)) * (inputMat - oldRangeBottom)) + newRangeBottom

def getX(slope, pt0, yFixed):
    '''
    (y - y0) = slope * (x - x0)
    x = ((y - y0) / slope) + x0
    '''
    return ((yFixed  - pt0[1]) / float(slope)) + pt0[0]
    
def getY(slope, pt0, xFixed):
    '''
    (y - y0) = slope * (x - x0)
    y = (slope * (x - x0)) + y0
    '''
    return (slope * (xFixed - pt0[0])) + pt0[1]
    
def getSlope(pt0, pt):
    '''
    (y - y0) = slope * (x - x0)
    slope = (y - yo) / (x - x0)
    
    Si el denominador va a ser 0 se divide entre un valor muy pequeno (1e-10).
    Asi se evita el mensaje de warning:
    'RuntimeWarning: divide by zero encountered in double_scalars'.
    '''
    if float(pt[0] - pt0[0]) == 0:
        return (pt[1] - pt0[1]) / 1e-10
    else:
        return (pt[1] - pt0[1]) / float(pt[0] - pt0[0])

def isSlopeHorizontal(slope):
    if slope >= -1 and slope < 1:
        return True
    else:
        return False

'''
Si dos rectas son perpendiculates tienen sus pendientes
inversas y cambiadas de signo.
'''
def getSlopePerp(slope):
    return -1 / float(slope)

'''
La funcion 'getCircumCentre' calcula el circuncentro de un triangulo definido
por tres puntos.

El circuncentro se calcula como la interseccion de tres rectas en las que cada una de ellas
pasa por la mitad de un lado formando un angulo recto con este.

En realidad, solo necesito calcular la interseccion de dos de esas tres rectas ya que
la tercera debe cruzar por esa misma interseccion.

              C
            /  \
         b /    \ a
          /      \
         /        \
     A  ----------- B
             c
'''
def getCircumCentre(ptA, ptB, ptC, verbose = False):
    '''
    y - yo = m * (x - x0)
    '''
    c2 = (((ptA[0] + ptB[0])/2.), (((ptA[1] + ptB[1])/2.)))
    cPerpSlope = getSlopePerp(getSlope(ptA, ptB))
    '''
    y - c2[1] = cPerpSlope * (x - c2[0])
    y = (cPerpSlope * (x - c2[0])) + c2[1]
    '''
    b2 = (((ptC[0] + ptA[0])/2.), (((ptC[1] + ptA[1])/2.)))
    bPerpSlope = getSlopePerp(getSlope(ptA, ptC))
    '''
    y - b2[1] = bPerpSlope * (x - b2[0])
    y = (bPerpSlope * (x - b2[0])) + b2[1]
    
    (cPerpSlope * (x - c2[0])) + c2[1] = (bPerpSlope * (x - b2[0])) + b2[1]
    (cPerpSlope * x) + (cPerpSlope * -c2[0]) + c2[1] = (bPerpSlope * x) + (bPerpSlope * -b2[0]) + b2[1]
    (cPerpSlope * x) - (bPerpSlope * x) = (bPerpSlope * -b2[0]) + b2[1] - (cPerpSlope * -c2[0]) - c2[1]
    x * (cPerpSlope - bPerpSlope) = (bPerpSlope * -b2[0]) + b2[1] - (cPerpSlope * -c2[0]) - c2[1]
    x = ((bPerpSlope * -b2[0]) + b2[1] - (cPerpSlope * -c2[0]) - c2[1]) / float(cPerpSlope - bPerpSlope)
    '''
    xCircumCtr = ((bPerpSlope * -b2[0]) + b2[1] - (cPerpSlope * -c2[0]) - c2[1]) / float(cPerpSlope - bPerpSlope)
    yCircumCtr = (cPerpSlope * (xCircumCtr - c2[0])) + c2[1]
    circumCtrA = np.sqrt(np.power((xCircumCtr - ptA[0]), 2) + np.power((yCircumCtr - ptA[1]), 2))
    circumCtrB = np.sqrt(np.power((xCircumCtr - ptB[0]), 2) + np.power((yCircumCtr - ptB[1]), 2))
    circumCtrC = np.sqrt(np.power((xCircumCtr - ptC[0]), 2) + np.power((yCircumCtr - ptC[1]), 2))
    if verbose == True:
        plt.figure(0, figsize = (15, 10))
        plt.title('circumcentre\ncircumCtrA = '+str(circumCtrA)+' | circumCtrB = '+str(circumCtrB)+' | circumCtrC = '+str(circumCtrC))
        plt.plot([ptA[0], ptB[0], ptC[0], ptA[0]], [ptA[1], ptB[1], ptC[1], ptA[1]], marker = 'o')
        plt.annotate('A', ptA, ptA)
        plt.annotate('B', ptB, ptB)
        plt.annotate('C', ptC, ptC)
        plt.scatter(c2[0], c2[1], marker = '*')
        plt.annotate('c2', c2, c2)
        plt.scatter(b2[0], b2[1], marker = '*')
        plt.annotate('b2', b2, b2)
        plt.scatter(xCircumCtr, yCircumCtr, marker = '^')
        plt.annotate('circumcentre', (xCircumCtr, yCircumCtr), (xCircumCtr, yCircumCtr))
        plt.show()
    return (xCircumCtr, yCircumCtr), circumCtrA
