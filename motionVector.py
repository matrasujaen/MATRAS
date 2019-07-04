'''
Created on 30/10/2015

@author: Grupo MATRAS. Universidad de Jaen.
'''
from matplotlib import pyplot as plt
import openpiv
import numpy as np
from scipy.cluster.vq import whiten, kmeans, vq
import cv2
from scipy.interpolate import griddata
import dataUtils

#===============================================================================
# CONSTANTES
#===============================================================================
funcDefStep = 0.5

#===============================================================================
# FUNCIONES
#===============================================================================
def kmHToPx(velKmH, intervalSec, resolKmPx):
    return ((velKmH/float(60 * 60)) * intervalSec)/float(resolKmPx)

def filterMotVectsByKmeans(u, v, verbose = False):
    '''
    La funcion 'filterMotVecByKmeans' utiliza la funcion 'kmeans' que emplea una funcion random por lo
    que esta funcion 'filterMotVecByKmeans' ejecutada varias veces con los mismos parametros puede dar resultados diferentes.
    '''
    '''
    La funcion 'whiten' de 'scipy.cluster.vq' normaliza un grupo de observaciones
    en base a sus caracteristicas.
    Antes de correr k-means es beneficioso reescalar cada dimension caracteristica
    del set de observacion con 'blanqueamiento'.
    Recibe como parametros:
        -obs    numpy array    Cada fila del array es una observacion. Las columnas son las caracteristicas vistas durante cada observacion.
    Devuelve como parametros:
        -result    numpy array    Contiene los valores de 'obs' escalados por la desviacion estandar de cada columna.
    Para mas informacion visitar el enlace:
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.whiten.html#scipy.cluster.vq.whiten
    '''
    obs = np.zeros((len(u), 2))
    obs[:, 0] = u[np.isnan(u) == False].copy()
    obs[:, 1] = v[np.isnan(v) == False].copy()
    obsNorm = whiten(obs)
    '''
    La funcion 'kmeans' de 'scipy.cluster.vq' realiza el k-means de un set de vectores de observacion formando k grupos.
    El algoritmo k-means ajusta los centroides hasta que el progreso suficiente no puede ser hecho, es decir, el cambio en
    la distorsion desde la ultima iteracion es menor que cierto umbral. Esto produce un 'libro de codigos' mapeando los centroides
    con los codigos y viceversa.
    La distorsion es definida como la suma de los cuadrados de las difrencias entre las observaciones y el centroide correspondiente.
    Recibe como parametros:
        -obs    numpy array    Cada fila del array de 'M' x 'N' es un vector de observacion. Las columnas son las caracteristicas
    vistas durante cada observacion. Las caracteristicas deben ser 'blanqueadas' primero con la funcion 'whiten'.
        -k_or_guess    int o numpy array    El numero de centroides a generar. Un codigo es asignado a cada centroide, que es tambien
    el indice de fila del centroide en la matriz de 'libro de codigos' generada. Los k centroides iniciales son escogidos
    al azar seleccionando observaciones de la matriz de observacion. Alternativamente, pasando un array de 'k' por 'N' se especifica
    los k centroides iniciales.
    Devuelve como parametros:
        -codebook    numpy array    Array de 'k' por 'N' de los k centroides. El i-esimo centroide de codebook[i] es representado
    con el codigo i. Los centroides y los codigos generados representan la menor distorsion vista, no necesariamente la minima
    distorsion global.
        -distortion    float    La distorsion entre las observaciones pasadas y los centroides generados.
    Para mas informacion visitar el enlace:
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html#scipy.cluster.vq.kmeans
    '''
    codeBook, distortion = kmeans(obsNorm, 2)
    '''
    La funcion 'vq' de 'scipy.cluster.vq' asigna un codigo de un 'libro de codigos' a cada observacion. Cada vector de observacion
    en el array 'obs' de 'M' x 'N' es comparado con los centroides en el 'libro de codigos' y se le asigna el codigo del centroide
    mas cercano.
    Recibe como parametros:
        -obs    numpy array    Cada fila del array de 'M' x 'N' es una observacion. Las columnas son las caracteristicas vistas
    durante cada observacion. Las caracteristicas deben ser 'blanquedas' primero usando la funcion 'whiten' o algo equivalente.
        -code_book    numpy array    El 'libro de codigos' es generado usualmente usando el algoritmo k-means. Cada fila del array
    contiene un codigo diferente, y las columnas son las caracteristicas del codigo.
    Devuelve como parametro:
        -code    numpy array    Un array de longitud 'M' que contiene los indices del 'libro de codigos' para cada observacion.
        -dist    numpy array    La distorsion (distancia) entre la observacion y su codigo mas cercano.
    Para mas informacion visitar el enlace:
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.vq.html#scipy.cluster.vq.vq
    '''
    code, dist = vq(obsNorm, codeBook)
    '''
    Como he impuesto k_or_guess = 2 centroides, 2 grupos, los valores de 'code' son ceros y unos.
    Asumo que el codigo que mas aparece es el correcto.
    '''
    num1s = code.sum()
    num0s = len(code) - num1s
    if num1s >= num0s:
        codeCorrecto = 1
    else:
        codeCorrecto = 0
    '''
    Finalmente devuelvo las observaciones originales tomadas como correctas, es decir,
    aquellas cuyo codigo asociado corresponde a 'codeCorrecto'.
    '''
    uKMeans = u[code == codeCorrecto]
    vKMeans = v[code == codeCorrecto]
    if verbose == True:
        plt.title('k-Means')
        plt.scatter(codeBook[:, 0], codeBook[:, 1], c = 'black')
        plt.annotate('centroid_0', (codeBook[0, 0], codeBook[0, 1]), (codeBook[0, 0], codeBook[0, 1]))
        plt.annotate('centroid_1', (codeBook[1, 0], codeBook[1, 1]), (codeBook[1, 0], codeBook[1, 1]))
        plt.scatter(obsNorm[:, 0][code == codeCorrecto], obsNorm[:, 1][code == codeCorrecto], c = 'green')
        plt.scatter(obsNorm[:, 0][code != codeCorrecto], obsNorm[:, 1][code != codeCorrecto], c = 'red')
        plt.show()
    return uKMeans, vKMeans

def filterMotVectsByModuleOutliers(u, v, verbose = False):
    uF = u.copy()
    vF = v.copy()
    modulos = np.sqrt(np.power(uF, 2) + np.power(vF, 2))
    outliersFilter = dataUtils.outlierFilter(modulos, maskValue = np.nan, verbose = verbose)
    uF[outliersFilter.mask == False] = np.nan
    vF[outliersFilter.mask == False] = np.nan
    return u, v

def getSingleMotVect(motVectsObjList, verbose = False):
    u = None
    v = None
    for motVectsObj in motVectsObjList:
        if u is None and v is None:
            u = motVectsObj.u.copy()
            v = motVectsObj.v.copy()
        else:
            u = np.hstack((u, motVectsObj.u.copy()))
            v = np.hstack((v, motVectsObj.v.copy()))
    '''
    Filtrado por modulo.
    Se eliminan aquellos vectores cuyo modulo sea detectado como
    un 'outlier'.
    '''
    u, v = filterMotVectsByModuleOutliers(u, v, verbose = verbose)
    u = u[np.isnan(u) == False]
    v = v[np.isnan(v) == False]
    '''
    Filtrado por k-Means.
    Se intuyen 2 grupos posibles:
        - Los vectores erroneos.
        - Los vectores correctos.
    Asumiendo que los correctos son los mas abundantes se excluyen los
    que se creen erroneos.
    '''
    u, v = filterMotVectsByKmeans(u, v, verbose = verbose)
    uMean = u.mean()
    vMean = v.mean()
    return (uMean, vMean)               

def getPtIndexInTraj(traj, pt):
    return np.sqrt(np.power(np.absolute(traj[:, 0] - pt[0]), 2) + np.power(np.absolute(traj[:, 1] - pt[1]), 2)).argmin()

def drawVectorFieldOnImage(img, x, y, u, v, intervalSec = 1):
    plt.imshow(img, interpolation = 'none', cmap = 'gray')
    plt.axis('off')
    '''
    Para pintar las flechas utilizaba plt.quiver() pero no me gusta que
    normalice los vectores.
    
    La funcion 'plt.arrow' pinta las flechas de una en una.
                        
    La funcion 'plt.arrow' pinta las flechas en base a los ejes que se hayan establecido.
    Si se hubiera puesto una/s flecha/s unicamente, la componente 'y' del grafico iria
    hacia arriba en el eje vertical.
    En este caso, los ejes los dispone el 'plt.imshow' previo, por lo que la componente
    'y' de los ejes crece hacia abajo.
                         
    El sentido de crecimiento de la componente 'y' es hacia arriba, al contrario que Python,
    por lo que la componente segun el eje que establece el 'plt.imshow' previo es hacia
    abajo, como en Python. Aqui no cabe cambiarle el signo porque no es un desplazamiento sino
    una posicion, asi que calculo la posicion segun el 'plt.imshow' que es el
    (numero_de_filas_de_la_imagen - 1) - componente_y.
    De esta forma, una componente 'y = 9' en una imagen con 10 filas, que seria la de arriba
    del todo, pasaria a ser la fila 0 segun el eje establecido por 'plt.imshow'.
                         
    Las unidades de cada componente 'u' y 'v' estan expresadas en pixeles/segundo por lo que
    para ver el desplazamiento de una imagen a otra las multiplico por los segundos
    transcurridos, quedando asi las unidades en pixeles.
                         
    Hay que definir el tamano de la cabeza de la flecha porque si no no se ve bien. Establezco
    el ancho de la cabeza de la flecha en el 1% del minimo de las dimensiones de la imagen.
    '''
    for fil in range(x.shape[0]):
        for col in range(x.shape[1]):
            if np.isnan(u[fil, col]) and np.isnan(v[fil, col]):
                plt.scatter(x[fil, col], ((img.shape[0] - 1) - y[fil, col]), s = 5, color = 'orange')
            else:
                plt.arrow(x[fil, col], ((img.shape[0] - 1) - y[fil, col]), u[fil, col] * intervalSec, -v[fil, col] * intervalSec, head_width = 0.01 * min(img.shape), color = 'green')

def getKernelEdges(fil, col, shape, winSize):
    if winSize < 3 or winSize % 2 != 1:
        return
    semiLenght = (winSize - 1)/2
    filPrev = max(0, (fil - semiLenght))
    filNext = min((fil + semiLenght), (shape[0] - 1))
    colPrev = max(0, (col - semiLenght))
    colNext = min((col + semiLenght), (shape[1] - 1))
    return filPrev, filNext, colPrev, colNext

def filterLonelyValues(u, v, winSize):
    if winSize < 3 or winSize % 2 != 1:
        return
    uLocal = u.copy()
    vLocal = v.copy()
    rows, cols = uLocal.shape
    for r in range(rows):
        for c in range(cols):
            if np.isnan(u[r, c]) == False and np.isnan(v[r, c]) == False:
                rowPrev, rowNext, colPrev, colNext = getKernelEdges(r, c, (rows, cols), winSize)
                ones = np.ones((((rowNext - rowPrev) + 1), ((colNext - colPrev) + 1)))
                windowU = u[rowPrev: (rowNext + 1), colPrev: (colNext + 1)].copy()
                windowV = v[rowPrev: (rowNext + 1), colPrev: (colNext + 1)].copy()
                numNans = ones[np.isnan(windowU) * np.isnan(windowV)].sum()
                if numNans > (ones.shape[0] * ones.shape[1])/2.:
                    uLocal[r, c] = np.nan
                    vLocal[r, c] = np.nan
    return uLocal, vLocal
            
#===============================================================================
# CLASES
#===============================================================================
'''
La clase 'openPiv' contiene los vectores de movimiento detectados entre dos imagenes. Y digo imagenes,
porque para que funcione correctamente, las matrices deben tener valores comprendidos entre 0 y 255.
Es muy dificil comprobar el correcto funcionamiento de esta clase ya que matrices amanadas dan resultados
erroneos. Por ello, para comprobar el correcto funcionamiento de esta clase lo mejor es repetir el
ejemplo contenido en el enlace:
    http://alexlib.github.io/openpiv-python/_downloads/tutorial-part1.zip

Los atributos x, y, u y v tienen el mismo tamano. Tanto x como u crecen hacia la derecha en el eje horizontal
y tanto y como v crecen hacia arriba en el eje vertical, al reves que Python. Por ello, si se quiere hacer uso
de estos atributos fuera de esta clase es necesario controlar la correcta interpretacion de los vectores.

No veo la implicacion que tiene el parametro 'intervalSec'.

En cuanto a pruebas, he comprobado que:
    + windowSize puede ser un valor impar, y, por lo tanto, no potencia de 2. Tiene que ser un valor positivo.
No da error si excede el tamano de la imagen.
    + overlapPx puede ser un valor impar, y, por lo tanto, no potencia de 2. Ademas puede ser positivo y exceder
el tamano de la imagen y no da error.
    + searchAreaSizePx puede ser inferior a windowSize y no tiene que ser ni par ni potencia de 2.

Para ejecutar una prueba descargar el paquete de:
    http://alexlib.github.io/openpiv-python/_downloads/tutorial-part1.zip
y utilizar sus imagenes para ejecutar el siguiente codigo:
import numpy as np
import openpiv.tools
rutaTutorial_part1 = '/home/javi/Descargas/tutorial-part1/tutorial-part1/'
frame_a = np.array(openpiv.tools.imread(rutaTutorial_part1+'exp1_001_a.bmp'), dtype = 'int32')
frame_b = np.array(openpiv.tools.imread(rutaTutorial_part1+'exp1_001_b.bmp'), dtype = 'int32')
movVect = openPiv(frame_a, frame_b, 0.02, 24, overlapPx = 12, searchAreaSizePx = 64, verbose = True)

He probado con una imagen de satelite y 32 me ha parecido un tamano de ventana bueno.
'''
class openPiv(object):
    def __init__(self, img2DPrev, img2DCurr, resolkmPx, intervalSec, windowSizePx = 32, overlapPx = None, searchAreaSizePx = None, sig2noiseMethod = 'peak2peak', maxVelKmH = 100, verbose = False):
        '''
        Existe un problema de "operacionalidad" de la clase que consiste en que, con imagenes
        muy grandes, se tarda mucho tiempo en realizar la funcion de correlacion cruzada aunque
        se reduzca el numero de vectores caracteristicos a obtener.
        Se penso en la posibilidad de reducir el tamano de la imagen pero finalmente se ha
        descartado porque complica mucho la comprension del metodo y se decide dejarlo
        fuera de la clase.
        '''
        self.img2DPrev = np.int32(img2DPrev.copy())
        self.img2DCurr = np.int32(img2DCurr.copy())
        self.intervalSec = intervalSec
        self.windowSizePx = int(round(windowSizePx))
        self.sig2noiseMethod = sig2noiseMethod
        self.maxVelKmH = maxVelKmH
        self.resolkmPx = resolkmPx
        '''
        Para los valores por defecto aplico la misma proporcion que lo que vienen en el enlace:
            http://www.openpiv.net/openpiv-python/src/generated/openpiv.process.extended_search_area_piv.html
        '''
        if overlapPx is None:
            self.overlapPx = int(round(self.windowSizePx/2.))
        else:
            self.overlapPx = int(round(overlapPx))
        if searchAreaSizePx is None:
            self.searchAreaSizePx = searchAreaSizePx
        else:
            self.searchAreaSizePx = int(round(searchAreaSizePx))
        '''
        La funcion 'openpiv.process.extended_search_area_piv' recibe los siguientes parametros:
            -frame_a    numpy array    Primer marco. 2d dtype = np.int32.
            -frame_b    numpy array    Segundo marco. 2d dtype = np.int32.
            -window_size    int    tamano de ventana de interrogacion cuadrada.
            -overlap    int    numero de pixeles en el que dos ventanas adyacentes se solapan.
            -dt    float    retardo de tiempo de separacion entre los dos frames.
            -search_area_size    int    tamano de ventana de interrogacion del segundo cuadro.
            -sig2Noise_method    string    OPT. define el metodo de medida del ratio senal a ruido.
        y devuelve los siguientes elementos:
            -u    numpy array    2d array que contiene la componente u de velocidad, en pixeles/segundos (entiendo que es segundo).
            -v    numpy array    2d array que contiene la componente v de velocidad, en pixeles/segundos (entiendo que es segundo).
            sig2noise    numpy array    OPT. 2d array que contiene el ratio senal a ruido de la funcion de correlacion cruzada.
        Para mas informacion visitar el enlace:
            http://www.openpiv.net/openpiv-python/src/generated/openpiv.process.extended_search_area_piv.html
        Dado que el resultado es en pixeles/segundo supongo que las unidades de 'dt' son segundos.
        Parece que las ventanas se plantean sobre la imagen de izquierda a derecha y de abajo a arriba.
        '''
        if self.searchAreaSizePx is not None:
            u, v, sig2noise = openpiv.process.extended_search_area_piv(self.img2DPrev, self.img2DCurr, window_size = self.windowSizePx, overlap = self.overlapPx, dt = self.intervalSec, search_area_size = self.searchAreaSizePx, sig2noise_method = self.sig2noiseMethod)
        else:
            '''
            La funcion 'openpiv.pyprocess.piv' es muy parecida a la ya comentada 'openpiv.process.extended_search_area_piv'.
            Algoritmo PIV estandar de correlacion cruzada.
            Es una pura implementacion en Python del algoritmo PIV estandar de correlacion cruzada. Es un predictor
            de desplazamiento de orden 0, y no se realiza un proceso iterativo.
            Para mas informacion visitar el enlace:
                http://www.openpiv.net/openpiv-python/src/generated/openpiv.pyprocess.piv.html#openpiv.pyprocess.piv
            Los argumentos de entrada y de salida son los mismos que para 'openpiv.process.extended_search_area_piv'
            a excepcion de la ausencia del argumento 'search_area_size'.
            Su funcionamiento se ha contrastado con el ejemplo de funcionamiento, dando un resultado que se
            se entiende como igual (aunque a nivel numerico de resultados no he comprobado si es exactamente igual,
            entenderia que no lo fuese ya que se trata de otro proceso diferente).
            '''
            u, v, sig2noise = openpiv.pyprocess.piv(self.img2DPrev, self.img2DCurr, window_size = self.windowSizePx, overlap = self.overlapPx, dt = self.intervalSec, sig2noise_method = self.sig2noiseMethod)
        '''
        La funcion 'openpiv.validation.sig2noise_val' reemplaza vectores falsos a traves del ratio senal
        a ruido de la correlacion cruzada.
        Reemplaza los vectores falsos con 0 si el ratio senal a ruido esta por debajo de un umbral.
        Recibe como parametros:
            -u    numpy array    2d array que contiene las componentes u de velocidad.
            -v    numpy array    2d array que contiene las componentes v de velocidad.
            -sig2noise    numpy array    2d array que contiene los valores del ratio
        senal a ruido de la funcion de correlacion cruzada.
            -threshold    float    valor umbral del ratio senal a ruido.
        Devuelve como parametros:
            -u    numpy array    2d array que contiene las componentes u de velocidad,
        donde los vectores falsos han sido reemplazados por Nan.
            -v    numpy array    2d array que contiene las componentes v de velocidad,
        donde los vectores falsos han sido reemplazados por Nan.
            -mask    numpy array    2d array buleano. Los valores 'True' corresponden a
        los valores atipicos.
        Para mas informacion visitar el enlace:
            http://www.openpiv.net/openpiv-python/src/generated/openpiv.validation.sig2noise_val.html
        Utilizo la media menos las desviacion tipica de los elementos finitos de 'sig2noise' como umbral.
        La componente 'v', segun los ejemplos, parece corresponder al eje vertical y ser positiva hacia
        arriba y negativa hacia abajo.
        La componente 'u', segun los ejemplo, corresponde al eje horizontal y es creciente hacia la derecha.
        '''
        '''
        Decido aplicar el filtro 'openpiv.validation.sig2noise_val' justo despues de calcular los vectores
        'u' y 'v' para que, a la entrada de esta funcion sigan manteniendo la forma de matriz bidimensional,
        como se indica en las instrucciones de la funcion.
        '''
        u, v, errorsMask = openpiv.validation.sig2noise_val(u, v, sig2noise, threshold = (sig2noise[np.isfinite(sig2noise)].mean() - sig2noise[np.isfinite(sig2noise)].std()))
        '''
        La funcion 'openpiv.validation.global_val' elimina vectores espureos con un umbral global.
        Estas pruebas de metodo de validacion para la consistencia espacial de los datos y vectores
        atipicos son remplazados con Nan (Not a Number) si al menos una de las dos componentes de
        velocidad esta fuera de un rango global especificado.
        
        Recibe como parametros:
            -u    numpy array    Array bidimensional que contiene la componente de velocidad 'u'.
            -v    numpy array    Array bidimensional que contiene la componente de velocidad 'v'.
            -u_thresholds    tupla    Tupla de dos elementos. u_thresholds = (u_min, u_max). Si
        u < u_min o u > u_max el vector es tratado como un valor atipico.
            -v_thresholds    tupla    Tupla de dos elementos. v_thresholds = (v_min, v_max). Si
        v < v_min o v > v_max el vector es tratado como un valor atipico.
        
        Devuelve como parametros:
            -u    numpy array    Array bidimensional que contiene la componente de velocidad 'u',
        donde los vectores espureos han sido remplazados por NaN.
            -v    numpy array    Array bidimensional que contiene la componente de velocidad 'v',
        donde los vectores espureos han sido remplazados por NaN.
            -mask    numpy array    Array bidimensional booleano. Los elementos True corresponden
        a valores atipicos.
        
        Para mas informacion visitar el enlace:
            http://www.openpiv.net/openpiv-python/src/generated/openpiv.validation.global_val.html
        '''
        despMaxPx = kmHToPx(self.maxVelKmH, self.intervalSec, self.resolkmPx)
        u, v, maxVelMask = openpiv.validation.global_val(u, v, ((-despMaxPx/float(self.intervalSec)), (despMaxPx/float(self.intervalSec))), ((-despMaxPx/float(self.intervalSec)), (despMaxPx/float(self.intervalSec))))
        '''
        La funcion 'openpiv.filters.replace_outliers' remplaza vectores invalidos en un campo de
        velocidad usando un algoritmo iterativo de pintado dentro de la imagen.
        El algoritmo es el siguiente:
            1. Para cada elemento en los arrays de componentes 'u' y 'v', se remplaza por un peso
        ponderado de los elementos vecinos que no son invalidos por ellos mismos. Los pesos
        dependen del tipo de metodo. Si "method = 'localmean'" los pesos son iguales a
        1/((2 * kernel_size + 1) ** 2 - 1).
            2. Varias iteraciones son necesarias si hay elementos invalidos adyacentes. Si este es
        el caso, la informacion es "extendida" desde los bordes de las regiones perdidas
        iterativamente, hasta que la variacion esta por debajo de un cierto umbral.
        
        Recibe como parametros:
            -u    numpy array    Array bidimensional de campo de componentes de velocidad 'u'.
            -v    numpy array    Array bidimensional de campo de componentes de velocidad 'v'.
            -max_iter    int    El numero de iteraciones.
            -kernel_size    int    El tamano del nucleo, que por defecto es 1.
            method    string    El tipo de nucleo usado para reparar vectores perdidos.
        
        Devuelve como parametros:
            -uf    numpy array    Array bidimensional del campo suavizado de componentes de
        velocidad 'u', donde los vectores invalidos han sido remplazados.
            -vf    numpy array    Array bidimensional del campo suavizado de componentes de
        velocidad 'v', donde los vectores invalidos han sido remplazados.
        
        Para mas informacion visitar el enlace:
            http://www.openpiv.net/openpiv-python/src/generated/openpiv.filters.replace_outliers.html
        '''
        u, v = openpiv.filters.replace_outliers(u, v, method = 'localmean', max_iter = 10)
        '''
        La funcion 'openpiv.process.get_coordinates' recibe los siguientes parametros de entrada:
            -image_size    tuple    tupla con el numero de filas y columnas.
            -window_size    int    tamano de la ventana de interrogacion.
            -overlapOverWSPerc    int    numero de pixeles en el que dos ventanas de interrogacion adyacentes se solapan.
        y devuelve los siguientes elementos:
            -x    numpy array    2d array que contiene las coordenadas x de los centros de las ventanas de interrogacion, en pixeles.
            -y    numpy array    2d array que contiene las coordenadas y de los centros de las ventanas de interrogacion, en pixeles.
        Para mas informacion visitar el siguiente enlace:
            http://www.openpiv.net/openpiv-python/src/generated/openpiv.process.get_coordinates.html
        Segun lo observado, los vectores 'u' y 'v' que genera openPiv se asocian a unas posiciones 'x' e 'y'. Ademas,
        se observa que, al igual que los vectores 'v', las posiciones 'y' crecen hacia arriba a lo largo del eje vertical.
        '''
        x, y = openpiv.process.get_coordinates(self.img2DPrev.shape, self.windowSizePx, self.overlapPx)
        '''
        Me he encontrado con que hay valores np.nan en ciertos elementos de 'u' y en ciertos elementos
        de 'v' pero ambas posiciones no coinciden. Debido a esto y a otras posibles "incoherencias" decido
        filtrar todos los atributos que se han calculado, verificando, de paso, que el tamano de los
        mismos es identico.
        '''
        uMask = np.isnan(u)
        vMask = np.isnan(v)
        xMask = np.isnan(x)
        yMask = np.isnan(y)
        '''
        Me da igual hacia donde se refiera la componente 'v', si deberia
        ser 5 o -5 porque al elevarla al cuadrado perdera el signo, pero
        para hacerlo correctamente le pongo un signo '-' porque crece hacia
        arriba en el eje vertical, al contrario que Python, por lo que le
        cambio el signo para que coincida con el sentido de crecimiento del
        eje vertical de Python (hacia abajo).
        Las unidades del modulo son tambien en px/seg, porque se calcula
        a partir de self.u y self.v, cuyas unidades son px/seg.
        '''
        modulos = np.sqrt(np.power(u, 2) + np.power((-v), 2))
        '''
        Filtro los modulos que de un frame a otro son menores de 1 pixel.
        '''
        modulosMask = np.zeros_like(modulos, dtype = np.bool)
        modulosMask[:, :] = False
        '''
        Cuando el filtro de eliminar componentes u/v con desplazamiento menor que 1 tras el tiempo
        de observacion se aplica y coinciden componentes u y v con valor np.nan
        aparece el warning:
            "RuntimeWarning: invalid value encountered in less"
        Sin embargo he observado el comportamiento y es el esperado.
        '''
        modulosMask[(np.absolute(u * self.intervalSec) < 1) & (np.absolute(v * self.intervalSec) < 1)] = True
        allFilters = np.invert(uMask) * np.invert(vMask) * np.invert(xMask) * np.invert(yMask) * np.invert(errorsMask) * np.invert(maxVelMask) * np.invert(modulosMask)
        '''
        Una vez que todos los calculos basicos para la obtencion de las variables de esta clase han sido
        realizados, filtro los valores invalidos manteniendo la forma de matriz de las variables.
        '''
        u[allFilters == False] = np.nan
        v[allFilters == False] = np.nan
        modulos[allFilters == False] = np.nan
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.modulos = modulos
        if verbose == True:
            plt.figure(figsize = (15, 10))
            if self.searchAreaSizePx is not None:
                plt.suptitle('openPiv vector\nintervalSec = '+str(self.intervalSec)+'[sec] | windowSizePx = '+str(self.windowSizePx)+'[px] | overlapPx = '+str(self.overlapPx)+'[px] | searchAreaSizePx = '+str(self.searchAreaSizePx)+'[px] | sig2noiseMethod = '+self.sig2noiseMethod)
            else:
                plt.suptitle('openPiv vector\nintervalSec = '+str(self.intervalSec)+'[sec] | windowSizePx = '+str(self.windowSizePx)+'[px] | overlapPx = '+str(self.overlapPx)+'[px] | sig2noiseMethod = '+self.sig2noiseMethod)
            plt.subplot2grid((2, 3), (0, 0))
            plt.title('previous image')
            plt.imshow(self.img2DPrev, interpolation = 'none', cmap = 'gray')
            plt.axis('off')
            plt.subplot2grid((2, 3), (1, 0))
            plt.title('next image')
            plt.imshow(self.img2DCurr, interpolation = 'none', cmap = 'gray')
            plt.axis('off')
            plt.subplot2grid((2, 3), (0, 1), colspan = 2, rowspan = 2)
            plt.title('openPiv vectors over next image')
            drawVectorFieldOnImage(self.img2DCurr, self.x, self.y, self.u, self.v, intervalSec = self.intervalSec)
            plt.show()
    
    def filterLonelyValues(self, verbose = True):
        if verbose == True:
            uOriginal = self.u.copy()
            vOriginal = self.v.copy()
        self.u, self.v = filterLonelyValues(self.u, self.v, 3)
        if verbose == True:
            plt.figure(0, figsize = (15, 7))
            plt.suptitle('filter lonely values')
            plt.subplot(121)
            plt.title('original')
            drawVectorFieldOnImage(self.img2DCurr, self.x, self.y, uOriginal, vOriginal, intervalSec = self.intervalSec)
            plt.subplot(122)
            plt.title('filtered')
            drawVectorFieldOnImage(self.img2DCurr, self.x, self.y, self.u, self.v, intervalSec = self.intervalSec)
            rows, cols = self.u.shape
            for r in range(rows):
                for c in range(cols):
                    if np.isnan(uOriginal[r, c]) == False and np.isnan(self.u[r, c]) == True and np.isnan(vOriginal[r, c]) == False and np.isnan(self.v[r, c]) == True:
                        plt.scatter(self.x[r, c], ((self.img2DCurr.shape[0] - 1) - self.y[r, c]), s = 5, color = 'red')
            plt.show()

'''
Tras la no exitosa experiencia de la implementacion del optical flow denso de Horn & Schunck
me propongo implementar una version esparcida del optical flow basado en el metodo elaborado por
Lucas-Kanade.
    https://es.wikipedia.org/wiki/Introducci%C3%B3n_al_flujo_%C3%B3ptico

Me basare en las explicaciones que se muestran en el siguiente enlace:
    http://docs.opencv.org/master/d7/d8b/tutorial_py_lucas_kanade.html#gsc.tab=0

El optical flow es el patron de movimiento aparente de objetos de imagenes entre dos cuadros (frames)
consecutivos causados por el movimiento del objeto o de la camara. Es un campo de vectores
bidimensional donde cada vector es un vector de desplazamiento que muestra el movimiento de puntos
desde el primer cuadro al segundo.
El optical flow trabaja sobre varios supuestos:
    1. Las intensidades de pixeles no cambian entre cuadros consecutivos.
    2. Pixeles vecinos tienen movimientos similares.

Los principios matematicos y las ecuaciones propuestas son las mismas que en el paper original de Horn
y Schunck, que se puede acceder a traves del sigiuente enlace:
    http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=1231385

No se puede resolver esta ecuacion con dos variables desconocidas. Asi que multiples metodos son propuestos
para resolver este problema y uno de ellos es Lucas-Kanade.

Como se ha visto en una suposicion anterior, todos los pixeles vecinos tendran un movimiento similar. El
metodo Lucas-kanade toma un parche de 3x3 alrededor del punto. Asi que todos los 9 puntos tienen el mismo
movimiento. Se puede encontrar el gradiente fx, fy, ft para estos 9 puntos. Asi que ahora el problema se
convierte en resolver 9 ecuaciones con dos variables desconocidas, por lo que el sistema de ecuaciones esta
sobredeterminado. Una mejor solucion es obtenida con un metodo de ajuste de minimos cuadrados.
(Se ofrece la semejanza entre la matriz inversa propuesta con el detector de esquinas de Harris. Ello denota
que las esquinas son los mejores puntos para ser seguidos.)

Asi desde el punto de vista del usuario, la idea es simple, se toman varios puntos a seguir, se reciben los
vectores de optical flow de esos puntos. Pero de nuevo hay algunos problemas. Hasta ahora, se ha tratado
con pequenos movimientos. Asi que se falla cuando es un movimiento grande. Asi que de nuevo se va
hacia piramides. Cuando se va hacia arriba en la piramide, los pequenos movimientos son eliminados y los
movimientos grandes se convierten en pequenos. Asi que aplicando Lucas-Kanade ahi, se obtiene el flujo
optico junto con la escala.

OpenCV provee todo esto en una unica funcion, 'cv2.calcOpticalFlowPyrLK()'. Para decidir los puntos, se
utiliza 'cv2.goodFeaturesToTrack()'. Para la funcion 'cv2.calcOpticalFlowPyrLK()' se le pasa el cuadro
anterior, los puntos anteriores y el cuadro siguiente. El devuelve los puntos siguientes junto con algunos
numeros de estado que tienen un valor de 1 si el siguiente punto es encotrado, y 0 para el resto.

Por defecto se inicializa el objeto para detectar un maximo de 100 esquinas a seguir, donde todas han de tener
como minimo (1/3.) de calidad que la mejor esquina hallada y una esquina de otra se distancia un 5% del lado
menor de la imagen inicial a procesar.

La eleccion de 'maxCorners = 1000' es porque pienso que 1000 puntos definirian bien el movimiento habido entre
dos imagenes.

No estoy seguro de si el numero de niveles de piramide me delimita el movimiento maximo y minimo a detectar. Creo
que si pero tambien creo que depende del tamano de la imagen a procesar.

Para comprobar la efectividad del metodo propongo el mismo codigo que se utiliza para
comprobar la efectividad del metodo OpenPIV:
import numpy as np
import openpiv.tools
rutaTutorial_part1 = '/home/javi/Descargas/tutorial-part1/tutorial-part1/'
frame_a = np.array(openpiv.tools.imread(rutaTutorial_part1+'exp1_001_a.bmp'), dtype = 'int32')
frame_b = np.array(openpiv.tools.imread(rutaTutorial_part1+'exp1_001_b.bmp'), dtype = 'int32')
movVect = sparseOpticalFlow(frame_a, frame_b, 0.02, 24, verbose = True)

He probado con una imagen de satelite y 13 me ha parecido un tamano de ventana bueno.

Tras hablar con Helio he dicidido aumentar el parametro 'qualityLevel' por defecto a
un valor '(20/100.)' con el objetivo de obtener puntos mas fiables en la funcion
'goodFeaturesToTrack' aunque se obtenga un menor numero de ellos.
'''
class sparseOpticalFlow(object):
    def __init__(self, img2DPrev, img2DCurr, resolkmPx, intervalSec, blockSizePx, windowSizePx = None, maxCorners = None, qualityLevel = (20/100.), minDistancePx = None, maxPirLevels = 2, maxVelKmH = 100, verbose = False):
        self.img2DPrev = np.uint8(img2DPrev.copy())
        self.img2DCurr = np.uint8(img2DCurr.copy())
        self.intervalSec = intervalSec
        self.blockSizePx = int(round(blockSizePx))
        self.maxVelKmH = maxVelKmH
        self.resolkmPx = resolkmPx
        '''
        Si no se indica parametro 'windowSizePx' se calcula un tamano de ventana
        tal que permita que un pixel del borde de la ventana pueda experimentar
        un desplazamiento maximo, dado el valor del parametro 'self.maxVelKmH'.
        '''
        if windowSizePx is None:
            self.windowSizePx = int(round(self.blockSizePx + (2 * kmHToPx(self.maxVelKmH, self.intervalSec, self.resolkmPx))))
        else:
            self.windowSizePx = int(round(windowSizePx))
        '''
        Si no se indica parametro 'maxCorners' se calcula cuantas esquinas se detectarian
        segun el tamano de ventana 'self.blockSizePx' si de cada ventana se obtuviese una esquina. 
        '''
        if maxCorners is None:
            self.maxCorners = int(np.floor(self.img2DPrev.shape[0]/float(self.blockSizePx) * self.img2DPrev.shape[1]/float(self.blockSizePx)))
        else:
            self.maxCorners = maxCorners
        self.qualityLevel = qualityLevel
        '''
        Si no se indica valor para minDistancePx se calcula
        como la mitad de 'self.blockSizePx'.
        '''
        if minDistancePx is None:
            self.minDistance = int(round(self.blockSizePx / 2.))
        else:
            self.minDistance = minDistancePx
        self.maxPirLevels = maxPirLevels
        '''
        La funcion 'goodFeaturesToTrack' determina esquinas fuertes en una imagen.
        Recibe como parametros:
            -image    Imagen de entrada monocanal de 8 bits o de tipo float de 32 bits.
            -maxCorners    Maximo numero de esquinas a devolver. Si hay mas esquinas que son
        halladas, las mas fuertes de ellas son devueltas.
            -qualityLevel    Parametro que caracteriza a la minima calidad aceptada de esquina de imagen.
        El valor del parametro es multiplicado por la mejor medida de calidad de esquina, que es el
        minimo 'eigenvalor' o la respuesta de la funcion de Harris. Las esquinas con la medida de calidad
        menor que el producto son rechazadas. Por ejemplo, si la mejor esquina tiene una medida de calidad
        de 1500, y 'qualityLevel = 0.01', entonces todas las esquinas con medida de calidad menor que 15
        son rechazadas.
            -mindistance    Minima distancia euclideana posible entre las esquinas devueltas.
            -mask    Region de interes opcional. Si la imagen no esta vacia (necesita tener el tipo
        CV_8UC1 y el mismo tamano que 'image'), especifica la region en la que las esquinas son detectadas.
            -blockSize    Tamano de un bloque medio para computarizar la matriz derivativa de covarianza
        sobre cada pixel del vecindario.
            -useHarrisDetector    Parametro que indica cuando usar el detector de Harris.
            -k    Parametro libre del detector de Harris.
        Devuelve:
            -corners    Vector de salida de esquinas detectadas.
        Para mas informacion visitar el siguiente enlace:
            http://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html
        He probado la funcion con una imagen artificial de 500 pixeles de ancho y 100 de alto.
        Esta imagen artificial es de fondo negro con dos rectangulos rojos perfectos. Los elementos
        se situan en relacion a las esquinas superiores izquierdas. Asi, un rectangulo de tamano 80
        pixeles de ancho por 50 de alto se situa en las coordenadas (30, 30) y el otro rectangulo de
        tamano 80 pixeles de ancho por 20 de alto si situa en las coordenadas (300, 30) (es decir,
        a la misma altura y desplazado hacia la derecha). Tambien se ha generado una segunda imagen
        artificial que es igual a la descrita pero con los elementos movidos 30 pixeles hacia la derecha
        y 10 pixeles hacia abajo.
        Al ejecutarse la funcion, entre las parejas de numeros que se obtienen como resultado
        hay valores en la primeras componentes que sobrepasasn los 100 pixeles, por lo que
        solo cabe que se trate de componentes del eje 'x' (eje horizontal) y por lo tanto,
        las restantes segundas componentes han de ser coordenadas 'y' (eje vertical).
        El resultado de la funcion 'goodFeaturesToTrack' (las coordenadas 'x' e 'y' de las
        esquinas) tiene forma de array de tantos elementos como esquinas haya detectado y,
        de cada elemento, tiene un array con dos componentes (coordenada 'x' y coordenada 'y').
        Dicho de otro modo, el resultado es un array de forma Nx1x2, donde 'N' es el numero de
        esquinas halladas.
        Ademas, el resultado al graficarse la ubicacion de las esquinas halladas sobre la imagen coincide
        perfectamente, se corrobora que se trata de coordenadas 'x' e 'y' y que la 'x' crece hacia
        la derecha en el eje horizontal y que 'y' crece hacia abajo en el eje vertical, como en Pyhton.
        
        Tras hablar con Helio he decidido hacer uso del parametro 'blockSize' para utilizarlo en el mismo
        sentido que el parametro 'window_size' en el metodo 'openPIV'.
        '''
        cornersPrev = cv2.goodFeaturesToTrack(self.img2DPrev, self.maxCorners, self.qualityLevel, self.minDistance, blockSize = self.blockSizePx)
        '''
        La funcion 'calcOpticalFlowPyrLK' calcula un optical flow para un set de caracteristicos dispersos 
        usando el metodo iterativo Lucas-Kanade con piramides.
        Recibe como parametros:
            -prevImg    Primera imagen de entrada de 8 bits o piramide construida con
        'buildOpticalFlowPyramid()'.
            -nextImg    Segunda imagen de entrada o piramide del mismo tamano y del mismo
        tipo que 'prevImg'.
            -prevPts    array bidimensional    Puntos en los que el flujo necesita ser hallado. Las
        coordenadas de puntos deben ser numeros en punto flotante de precision simple.
            -nextPts    array bidimensional    Vector de salida de puntos en dos dimensiones (con
        coordenadas en punto flotante de precision simple) que contiene las nuevas posiciones calculadas
        de los caracteristicos de entrada en la segunda imagen; cuando el flag 'OPTFLOW_USE_INITIAL_FLOW'
        es pasado, el vector debe tener el mismo tamano que en la entrada.
            -status    array    Vector de salida de estado (de chars sin signo); cada elemento del vector
        es puesto a 1 si el flujo de los caracteristicos correspondientes ha sido hallado, de cualquier otro
        modo, es puesto a 0.
            -err    array    Vector de salida de errores; cada elemento del vector es puesto a un error
        para el caracteristico correspondiente, el tipo de error de medida puede ser establecido en los
        flags de parametros; si el flujo no se encontro entonces el error no es definido (usa los
        parametros de 'status' para encontrar tales casos).
            -winSize    Tamano de la ventana de busqueda en cada nivel de piramide.
            -maxLevel    Numero de maximo nivel de piramide de base 0. Si es puesto a 0, las piramides no son usadas
        (nivel unico), si es puesto a 1, son usados dos niveles, y en adelante; si las piramides son
        pasadas como input entonces el algoritmo usara tantos niveles como tengan las piramides pero no
        mas de 'maxlevel'.
            -criteria    Parametro, especificando el criterio de terminacion del algoritmo iterativo de busqueda
        (tras el numero maximo especificado de iteraciones 'criteria.maxCount' o cuando la
        ventana de busqueda se mueve menos que 'criteria.epsilon').
            -flags    Flags de operacion. Pueden ser:
                + OPTFLOW_USE_INITIAL_FLOW    utiliza estimaciones iniciales, almacenadas en 'nextPts';
        si el flag no es puesto, entonces 'prevPts' es copiado a 'nextPts' y es considerado
        la estimacion inicial.
                + OPTFLOW_LK_GET_MIN_EIGENVALS    usa el minimo 'eigen-valor' como medida de error; si
        el flag no es puesto, entonces la distancia L1 entre parches alrededor del original y un punto
        movido, dividido por el numero de pixeles en una ventana, es usado como medida de error.
            -minEigThreshold    El algoritmo calcula el minimo 'eigen valor' de una matriz normal de 2x2
        de ecuaciones del optical flow, dividida por el numero de pixeles en una ventana; si este valor
        es menor que 'minEigThreshold', entonces un caracteristico correspondiente es apartado y su
        flujo no es procesado, asi que esto permite eliminar malos puntos y obtener un aumento del
        rendimiento. 
        Para mas informacion visitar el enlace:
            http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
        '''
        nextPts, status, err = cv2.calcOpticalFlowPyrLK(self.img2DPrev, self.img2DCurr, cornersPrev, winSize = (self.windowSizePx, self.windowSizePx), maxLevel = self.maxPirLevels)
        '''
        Al filtrar los vectores de puntos por su estado se pierde una dimension.
        '''
        cornersPrevFinal = cornersPrev[status == 1]
        xList = cornersPrevFinal[:, 0]
        '''
        Para que coincida con el OpenPIV, las coordenadas 'y' crecen hacia arriba en el eje vertical.
        '''
        yList = (self.img2DPrev.shape[0] - 1) - cornersPrevFinal[:, 1]
        cornersNextFinal = nextPts[status == 1]
        error = err[status == 1]
        uTotal = cornersNextFinal[:, 0] - cornersPrevFinal[:, 0]
        '''
        multiplico por '-1' la diferencia de coordenadas de puntos de esquinas para que,
        cumpliendo con el estandar en estos casos, la coordenada del eje vertical crezca
        hacia arriba, al reves que Python.
        '''
        vTotal = -(cornersNextFinal[:, 1] - cornersPrevFinal[:, 1])
        uList = uTotal/float(self.intervalSec)
        vList = vTotal/float(self.intervalSec)
        despMaxPx = kmHToPx(self.maxVelKmH, self.intervalSec, self.resolkmPx)
        uList, vList, maskMaxVel = self.global_val(uList, vList, ((-despMaxPx/float(self.intervalSec)), (despMaxPx/float(self.intervalSec))), ((-despMaxPx/float(self.intervalSec)), (despMaxPx/float(self.intervalSec))))
        '''
        Le aplico el mismo filtro que utilice en la clase 'openPiv' para eliminar
        aquellas muestras que representan movimientos menores a un pixel entre un
        frame y el siguiente.
        '''
        modulosList = np.sqrt(np.power((uList * self.intervalSec), 2) + np.power((-vList * self.intervalSec), 2))
        modulosMask = np.zeros_like(modulosList, dtype = np.bool)
        modulosMask[:] = False
        modulosMask[(np.absolute(uList * self.intervalSec) < 1) & (np.absolute(vList * self.intervalSec) < 1)] = True
        allFilters = np.invert(error > (error.mean() + error.std())) * np.invert(maskMaxVel) * np.invert(modulosMask)
        uList[allFilters == False] = np.nan
        vList[allFilters == False] = np.nan
        '''
        He comprobado que las posiciones con valor np.nan en 'uList'
        son las mismas que las posiciones con valor np.nan en 'vList',
        como era de esperar.
        '''
        check = np.isnan(vList) == np.isnan(uList)
        if False in check:
            return
#         xList = xList[allFilters == True]
#         yList = yList[allFilters == True]
        '''
        Se necesita que, al igual que con la clase 'openPiv', tanto las componentes 'u' y 'v'
        como las coordenadas 'x' e 'y' se hallen representados sobre la imagen en forma
        de matriz equiespaciada.
        Se vuelve a usar la funcion 'openpiv.process.get_coordinates' para obtener
        el punto central que corresponde a cada ventana de analisis.
        '''
        self.x, self.y = openpiv.process.get_coordinates(self.img2DPrev.shape, self.blockSizePx, 0)
        self.u = np.zeros_like(self.x)
        self.u[:, :] = np.nan
        self.v = np.zeros_like(self.x)
        self.v[:, :] = np.nan
        self.modulos = np.zeros_like(self.x)
        self.modulos[:, :] = np.nan
        '''
        La variable 'shareCell' es un diccionario cuya clave es una tupla
        con los valores de (fila, columna) de 'self.u' o de 'self.v' (porque
        la forma es la misma) y el valor es un numpy array de dos columnas:
        componente 'u' y componente 'v'. Este numpy array tiene tantas filas
        como componentes hayan querido asignarse a la misma celda de
        'self.u'/'self.v'.
        '''
        shareCell = {}
        for contItem in range(len(xList)):
            '''
            Esta variable 'fila' tiene el sentido creciente hacia arriba en el eje vertical,
            al contrario que Python.
            '''
            fila = int(yList[contItem] / self.blockSizePx)
            col = int(xList[contItem] / self.blockSizePx)
            '''
            Como la asignacion por filas y columnas sigue el sentido de crecimiento de Python
            en el momento de asignar las filas debo de tener en cuenta que quiero un sentido
            de crecimiento hacia arriba, al reves que Python, por lo que el indice de fila
            lo calculo como el (total_de_filas - 1) - fila_que_deseo_asignar. 
            
            En el ajuste a grid hay muestras (las de los bordes) que pueden no tenerse en
            cuenta porque no hay celda que las abarque. 
            '''
            if (np.isnan(uList[contItem]) == False) and (fila <= (self.y.shape[0] - 1)) and (col <= (self.x.shape[1] - 1)):
                if np.isnan(self.u[((self.y.shape[0] - 1) - fila), col]):
                    self.u[((self.y.shape[0] - 1) - fila), col] = uList[contItem]
                    self.v[((self.y.shape[0] - 1) - fila), col] = vList[contItem]
                else:
                    '''
                    Si se mete por aqui se mete tambien en el sitio correspondiente a la
                    componente 'v'. Lo he comprobado.
                    
                    He comprobado que el numero de vectores finales tras ajustar estos
                    al grid equiespaciado (un vector por ventana) es diferente del numero
                    de vectores antes de hacer el ajuste al grid. Esto es debido a que varios
                    vectores inicialmente calculados se quieren meter en la misma celda del
                    grid equiespaciado. Lo que a su vez significa que este metodo Optical Flow
                    puede obtener mas de un punto a seguir por ventana de analisis.
                    La comprobacion de estos hechos se ha visto midiendo que:
                        - El numero de vectores inicialmente calculados con valor diferente a 'np.nan'
                    (tras haber sido filtrados) era de 210.
                        - El numero de referencias a celdas del grid anteriormente inicializadas
                    es de 64.
                        - En el grid equiespaciado existen 146 vectores con valor distinto a 'np.nan'.
                    Si se suman los 146 vectores finales del grid equiespaciado + los 64 vectores que
                    han compartido celda en el grid equiespaciado dan como resultado los 210 vectores
                    con valor distinto de 'np.nan' inicialmente obtenidos.
                    
                    Para solventar este hecho lo mejor es promediar el valor entre todos los vectores
                    que se refieren a la misma celda del grid equiespaciado.
                    '''
                    if (((self.y.shape[0] - 1) - fila), col) in shareCell.keys():
                        shareCell[(((self.y.shape[0] - 1) - fila), col)] = np.vstack((shareCell[(((self.y.shape[0] - 1) - fila), col)], np.array([uList[contItem], vList[contItem]])))
                    else:
                        '''
                        Si entra por esta rama es porque es la primera vez que se repite la asignacion
                        a una celda de 'self.u'/'self.v', luego ya existe un valor valido en esa
                        celda. Lo que se hace el valor de esa celda de las variables 'self.u' y
                        'self.v', almacenarlo en el diccionario 'shareCell' y sustituir en valor de
                        la celda en 'self.u'/'self.v' por np.nan.
                        '''
                        shareCell[(((self.y.shape[0] - 1) - fila), col)] = np.vstack((np.array([self.u[(((self.y.shape[0] - 1) - fila), col)], self.v[(((self.y.shape[0] - 1) - fila), col)]]), np.array([uList[contItem], vList[contItem]])))
                        self.u[(((self.y.shape[0] - 1) - fila), col)] = np.nan
                        self.v[(((self.y.shape[0] - 1) - fila), col)] = np.nan
                self.modulos[fila, col] = modulosList[contItem]
        for key in shareCell.keys():
            self.u[key] = shareCell[key][:, 0].mean()
            self.v[key] = shareCell[key][:, 1].mean()
        if verbose:
            plt.figure(figsize = (15, 10))
            plt.suptitle('Lukas-Kanade sparse optical flow\nintervalSec = '+str(self.intervalSec)+'[sec] | maxCorners = '+str(self.maxCorners)+' | qualityLevel = '+str(round(self.qualityLevel, 3))+' | minDistance = '+str(round(self.minDistance, 3))+'[px]\nblockSizePx = '+str(self.blockSizePx)+'[px] |  windowSizePx = '+str(round(self.windowSizePx, 3))+'[px] | maxPirLevels = '+str(self.maxPirLevels))
            plt.subplot2grid((3, 3), (0, 0))
            plt.title('previous image corners')
            plt.imshow(self.img2DPrev, interpolation = 'none', cmap = 'gray')
            plt.axis('off')
            contPt = 0
            for contItem in range(len(xList)):
                if not np.isnan(uList[contItem]):
                    plt.scatter(xList[contItem], ((self.img2DCurr.shape[0] - 1) - yList[contItem]), color = 'red')
                    plt.annotate(str(contItem), (xList[contItem], ((self.img2DCurr.shape[0] - 1) - yList[contItem])), (xList[contItem], ((self.img2DCurr.shape[0] - 1) - yList[contItem])), color = 'red')
                    contPt += 1
            plt.subplot2grid((3, 3), (0, 1))
            plt.title('next image corners')
            plt.imshow(self.img2DCurr, interpolation = 'none', cmap = 'gray')
            plt.axis('off')
            contPt = 0
            for contItem in range(len(xList)):
                if not np.isnan(uList[contItem]):
                    plt.scatter((xList[contItem] + (uList[contItem] * self.intervalSec)), ((self.img2DCurr.shape[0] - 1) - (yList[contItem] + (vList[contItem] * self.intervalSec))), color = 'green')
                    plt.annotate(str(contItem), ((xList[contItem] + (uList[contItem] * self.intervalSec)), ((self.img2DCurr.shape[0] - 1) - (yList[contItem] + (vList[contItem] * self.intervalSec)))), ((xList[contItem] + (uList[contItem] * self.intervalSec)), ((self.img2DCurr.shape[0] - 1) - (yList[contItem] + (vList[contItem] * self.intervalSec)))), color = 'green')
                    contPt += 1
            plt.subplot2grid((3, 3), (0, 2))
            plt.title('optical flow vectors field over next image')
            plt.imshow(self.img2DCurr, interpolation = 'none', cmap = 'gray')
            plt.axis('off')
            contPt = 0
            for contItem in range(len(xList)):
                if not np.isnan(uList[contItem]):
                    plt.arrow(xList[contItem], ((self.img2DCurr.shape[0] - 1) - yList[contItem]), uList[contItem] * self.intervalSec, -vList[contItem] * self.intervalSec, head_width = 0.01 * min(img2DCurr.shape), color = 'green')
                    contPt += 1
            plt.subplot2grid((3, 3), (1, 0), rowspan = 2, colspan = 3)
            plt.title('optical flow grid over next image')
            drawVectorFieldOnImage(self.img2DCurr, self.x, self.y, self.u, self.v, intervalSec = self.intervalSec)
            plt.show()
    '''
    La funcion 'global_val' emula el comportamiento de la funcion
    'openpiv.validation.global_val' que se utiliza en la clase 'openPiv'
    '''
    def global_val(self, u, v, (u_min, u_max), (v_min, v_max)):
        uLocal = u.copy()
        vLocal = v.copy()
        mask = np.zeros(len(uLocal), dtype = np.bool)
        mask[:] = False
        uLocal[(uLocal < u_min) | (uLocal > u_max)] = np.nan
        vLocal[(vLocal < v_min) | (vLocal > v_max)] = np.nan
        mask[np.isnan(uLocal)] = True
        mask[np.isnan(vLocal)] = True
        return uLocal, vLocal, mask
    
    def filterLonelyValues(self, verbose = False):
        if verbose == True:
            uOriginal = self.u.copy()
            vOriginal = self.v.copy()
        self.u, self.v = filterLonelyValues(self.u, self.v, 3)
        if verbose == True:
            plt.figure(0, figsize = (15, 7))
            plt.suptitle('filter lonely values')
            plt.subplot(121)
            plt.title('original')
            drawVectorFieldOnImage(self.img2DCurr, self.x, self.y, uOriginal, vOriginal, intervalSec = self.intervalSec)
            plt.subplot(122)
            plt.title('filtered')
            drawVectorFieldOnImage(self.img2DCurr, self.x, self.y, self.u, self.v, intervalSec = self.intervalSec)
            rows, cols = self.u.shape
            for r in range(rows):
                for c in range(cols):
                    if np.isnan(uOriginal[r, c]) == False and np.isnan(self.u[r, c]) == True and np.isnan(vOriginal[r, c]) == False and np.isnan(self.v[r, c]) == True:
                        plt.scatter(self.x[r, c], ((self.img2DCurr.shape[0] - 1) - self.y[r, c]), s = 5, color = 'red')
            plt.show()

'''
La clase 'subSampledMotVectField' submuestrea un campo de movimiento.
'''
class subSampledMotVectField(object):
    def __init__(self, motVectObj, subSampleWinSizePx = 1):
        '''
        'self.subSampleWinSizePx' representa por cada cuantos pixeles hay un
        valor submuestreado.
        '''
        self.img2DPrev = motVectObj.img2DPrev
        self.img2DCurr = motVectObj.img2DCurr
        self.intervalSec = motVectObj.intervalSec
        self.subSampleWinSizePx = subSampleWinSizePx
        self.x, self.y = openpiv.process.get_coordinates(motVectObj.img2DCurr.shape, self.subSampleWinSizePx, 0)
        '''
        La funcion 'griddata' interpola datos de D dimensiones no estructurados.
        Recibe como parametros:
            -points    ndarray of floats    datos de puntos de coordenadas.
            -values    ndarray of float    datos de valores.
            -xi    ndarray of float    puntos en los que interpolar los datos.
            -method    optional    metodo de interpolacion (linear, nearest, cubic)
        Para mas informacion visitar el enlace:
            http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.griddata.html
        
        La funcion 'np.c_' traduce objetos rebanada a una concatenacion a lo largo del segundo eje.
        La funcion 'np.c' apila vectores rebanada en columnas (de ahi la 'c').
        Hay otra funcion que se llama np.r que apila los arrays por filas (de ahi la 'r' de rows).
        Para mas informacion visitar el enlace:
            http://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html
        
        El uso combinado de ambas funciones consigue que:
            - las coordenadas que representan a una matriz de datos se aplanen y se unan a lo largo
            de un eje por lo que el compuesto sigue teniendo los mismos elementos que la matriz a la
            que representan (llamemosla 'x' que en el ejemplo tendria 3 elementos).
            a = 1,
                2,
                3
            b = 4,
                5,
                6
            c = np.c_[a, b]
            c = [1  [4
                 2   5
                 3]  6]
            'c' conforma el parametro 'points' de la funcion 'griddata' mientras que 'x' es el
            parametro 'values' de dicha funcion. El parametro 'xi' de la funcion 'griddata'
            seran las nuevas matrices a las que se quiere interpolar 'values'.
            - El resultado que se devuelve sera 'values' pero relacionado ahora por los elementos
            de las matrices 'xi'.
        
        Viendo el comportamiento de la funcion 'griddata' parece que cuando entra en el promediado
        algun valor 'np.nan' el resultado es 'np.nan'. es por ello que se debe excluir del
        promediado los valores 'np.nan'.
        
        Como las componentes de movimiento 'u' y 'v' provienen de un objeto de calculo de movimiento,
        se entiende que los valores nulos de cada componente 'u' y 'v' se encuentran en las mismas
        posiciones.
        '''
        nullMask = np.isnan(motVectObj.u) & np.isnan(motVectObj.v)
        self.u = griddata(np.c_[motVectObj.x[nullMask == False].flatten(), motVectObj.y[nullMask == False].flatten()], motVectObj.u[nullMask == False].flatten(), (self.x.flatten(), self.y.flatten()), method='linear')
        self.v = griddata(np.c_[motVectObj.x[nullMask == False].flatten(), motVectObj.y[nullMask == False].flatten()], motVectObj.v[nullMask == False].flatten(), (self.x.flatten(), self.y.flatten()), method='linear')
        self.u = np.reshape(self.u, self.x.shape)
        self.v = np.reshape(self.v, self.x.shape)

'''
La clase 'flow' convierte un campo de vectores equiespaciado
en lineas de corriente. Cada linea de corriente la forman puntos y
cada de cada punto hay informacion de sus coordenadas 'x', 'y' y
velocidad, en este orden.
'''
class movFlow(object):
    def __init__(self, motVectObj, density = 1, fill = True, subSampleWinSizePx = None, verbose = False):
        self.density = density
        self.fill = fill
        self.subSampleWinSizePx = subSampleWinSizePx
        '''
        De cara a obtener un flujo mejor definido necesito tener un campo
        de vectores (correcto) lo mas denso posible. Es por ello que realizo
        dos procesos:
            1. Submuestreo de vectores.
            2. Rellenado de vectores nulos.
        Realizo, por tanto, los dos procesos:
        1. Submuestreo de vectores.
        '''
        if self.subSampleWinSizePx is not None:
            motVect = subSampledMotVectField(motVectObj, self.subSampleWinSizePx)
        else:
            motVect = motVectObj
        '''
        2. Rellenado de vectores nulos.
        '''
        if self.fill == True:
            self.u, self.v = self.fillNullValues(motVect)
        else:
            self.u = motVect.u.copy()
            self.v = motVect.v.copy()
        '''
        Quiero determinar el grosor de las lineas en funcion del modulo del campo de
        vectores en un rango de 1 a 5. 
        '''
        self.modules = np.sqrt(np.power(self.u, 2) + np.power(self.v, 2))
        '''
        Para evitar warnings en 'dataUtils.changeRange' me aseguro de que 'self.modules'
        contenga valores diferentes.
        '''
        if self.modules.min() == self.modules.max():
            lw = np.zeros_like(self.modules) + ((1 + 5) / 2.)
        else:
            lw = dataUtils.changeRange(self.modules, self.modules.min(), self.modules.max(), 1, 5)
        '''
        La funcion 'matplotlib.pyplot.streamplot' dibuja las lineas de corriente
        de un flujo de vectores.
        
        Recibe como parametros:
            -x, y    array    Array unidimensional que define una cuadricula
        uniformemente espaciada. Se trata de un array con las coordenadas 'x'
        e 'y', respectivamente, del campo de vectores.
            -u, v    array    Velocidades en 'x' e 'y'. El numero de filas
        debe coincidir con la longitud de 'y', y el numero de columnas debe
        coincidir con el de 'x'.
            -density:    float o tupla de 2 elementos    Controla la proximidad de
        las lineas de corriente. Cuando 'density = 1', el dominio es dividido en un
        grid de 30x30 (la densidad escala linealmente este grid). Cada celda en el grid
        puede tener, como mucho, una linea de corriente atravensandola. Para diferentes
        densidades en cada direccion, usa '[densidad_x, densidad_y]'.
            -lineWidth    numerico o 2D array    Varia el ancho de linea cuando
        se da un array bidimensional de la misma dimension que las velocidades.
            -color    codigo de color de matplotlib o 2D array    Color de las
        lineas de corriente. Cuando se da un array con la misma dimension que las velocidades,
        los valores de 'color' son convertidos a colores usando 'cmap'.
            -cmap    Colormap    Colormap usado para imprimir las lineas de corriente y flechas.
        Solo es necesario cuando se usa un array como entrada para 'color'.
            -norm    Normalize    Normaliza el objeto usado a una escala de datos de luminancia
        de 0 a 1. Si es 'None', se extiende (min, max) a (0, 1). Solo es necesario cuando 'color' es
        un array.
            ...
        
        Devuelve como resultado:
            -stream_container    StreamplotSet    Objeto contenedor con atributos:
                +lines    matplotlib.collections.LineCollection    Lineas de corriente.
                +arrows    matplotlib.patches.FancyArrowPatch    Coleccion de objetos representando
            flechas en mitad de las lineas de corriente.
        
        Para mas informacion visitar el enlace:
            http://matplotlib.org/api/pyplot_api.html
        
        Observaciones sobre la funcion 'matplotlib.pyplot.streamplot':
            -Cuando se le pasan variables creadas en sentido de Python se devuelve un
        grafico donde las componentes 'v' e 'y' crecen hacia arriba en el eje vertical.
            -Cuando se le pasan las mismas variables que en el caso anterior y se imprime
        antes una imagen se invierte el sentido de crecimiento de las componentes 'v' e 'y'
        por lo que se entiende que adquiere los ejes de la imagen.
        '''
        streamPlot = plt.streamplot(motVect.x[0, :], motVect.img2DCurr.shape[0] - 1 - motVect.y[:, 0], self.u, -self.v, density = self.density, color = self.modules, linewidth = lw, cmap = 'YlOrRd')
        '''
        'matplotlib.pyplot.streamplot' crea un grafico que es necesario cerrar para que
        no aparezca en posteriores graficos.
        '''
        plt.close()
        '''
        Cada elemento de 'segments' son dos puntos definidos en coordenadas 'x' e 'y'.
        '''
        segments = streamPlot.lines.get_segments()
        '''
        El numero de elementos de 'segments' es el mismo que el numero de
        elementos que devuelve su metodo 'get_linewidth()' que pertenece
        al atributo 'lines' de 'plt.streamplot'por lo que cabe pensar
        que existe un ancho de linea asociado a cada segmento, lo que encaja,
        ademas, con la observacion visual.
        
        'self.lineWidths' debe tener relacion con 'lw' pero evito pasos
        intermedios y tomo los grosores de 'plt.streamplot.lines.get_linewidth()'.
        '''
        self.lineWidths = streamPlot.lines.get_linewidth()
        '''
        Los colores, accesibles a traves del metodo 'matplotlib.pyplot.streamplot.lines.get_color()',
        no los puedo conocer hasta que el figure no se imprime por pantalla por lo que renuncio a
        utilizarlos. En cualquier caso, su finalidad es estetica ya que la informacion de velocidad
        de cada segmento perteneciente a una trayectoria la extraigo de su grosor asociado.
        '''
        '''
        Cada elemento de 'arrowPaths' es un objeto 'matplotlib.path.Path'
        que representa una serie de segmentos de lineas y curvas que pueden
        ser desconectados o cerrados. El contenido se compone de dos numpy
        arrays paralelos: vertices (atributo 'vertices') y codigos (atributo
        'codes').
        
        La funcion de cada vertice se conoce por su codigo asociado.
        Los codigos son:
            0 - (1 vertice) stop
            1 - (1 vertice) muevete a
            2 - (1 vertice) dibuja una linea desde la posicion actual al vertice dado.
            3 - (1 punto de control, 1 punto final) dibuja una curva bezier cuadratica
        desde la posicion actual, con el punto de control dado, hasta el punto final dado.
            4 - (2 puntos de control, 1 punto final) dibuja una curva bezier cubica desde
        la posicion actual, con los puntos de control dados, hasta el punto final dado.
            79 - (1 vertice) dibuja un segmento de linea hasta el punto de inicio de la
        actual polilinea.
        
        Para mas informacion sobre este tipo de objetos visitar el enlace:
            http://matplotlib.org/api/path_api.html
        Basandome en los ejemplos hasta ahora vistos se observa que:
            -Los codigos de las flechas parecen tener todos la misma estructura:
                indice: [ 0  1  2  3  4  5  6  7]
                codes:  [ 1  3  3  1  2  2  2 79] 
            -Los vertices penultimo y en la ultimo tienen el mismo valor.
            -Las trayectorias se dibujan desde la primera posicion hasta la
            ultima indicando asi el sentido de avance de la trayectoria.
        
        Finalmente no utilizo las flechas, accesibles a traves de
        'matplotlib.pyplot.streamplot.arrows.properties()['paths']' porque no se que relacion
        tienen con las trayectorias.
        ''' 
        '''
        Me propongo separar todas los segmentos en trayectorias.
        Cada vez que el principio de un nuevo segmento no corresponda
        al final del anterior es que este nuevo segmento corresponde
        a otra trayectoria.
        
        Asi, cada trayectoria esta formada por uno o varios segmentos
        y cada segmento esta formado por dos puntos (definido cada punto
        por dos coordenadas) y su ancho correspondiente.
        
        Inicializo el diccionario de trayectorias 'trajectories' con el primer
        elemento.
        '''
        trajectories = {}
        contKey = 0
        trajectories[contKey] = [segments[0]], [self.lineWidths[0]]
        for contSeg in range(1, len(segments)):
            if False in (segments[contSeg][0] == segments[contSeg-1][1]):
                '''
                El segmento corresponde a otra trayectoria.
                '''
                contKey += 1
                trajectories[contKey] = [segments[contSeg]], [self.lineWidths[contSeg]]
            else:
                '''
                El segmento corresponde a la misma trayectoria.
                '''
                trajectories[contKey][0].append(segments[contSeg])
                trajectories[contKey][1].append(self.lineWidths[contSeg])
        '''
        Me propongo calcular la sucesion de puntos de cada trayectoria. Para ello
        sigo el esquema interno de 'trajectories' solo que en lugar de dos puntos
        por segmento quiero tener todos los puntos que forman el segmento con una
        definicion de 1 [px].
        Asi, cada trayectoria se definira por puntos (de dos coordenadas cada uno,
        'x' e 'y') y cada punto tendra asociada una velocidad (ahora en terminos
        de velocidad, no de grosor). 
        '''
        self.trajsPts = {}
        for key in trajectories.keys():
            firstPt = True
            for contSeg in range(len(trajectories[key][0])):
                if False in (trajectories[key][0][contSeg][0] == trajectories[key][0][contSeg][1]):
                    '''
                    Verifico que los puntos que definen el segmento son diferentes
                    porque, en ocasiones, son el mismo.
                    Puede ser que el primer punto y el segundo (y puede que el tercero, cuarto, etc.)
                    sean el mismo, por lo que, de ser asi, el primer punto de la trayectoria ('contSeg = 0')
                    no se almacenaria. Es por ello que en lugar de utilizar el contador 'contSeg' para comprobar
                    de si se trata del primer punto de la trayectoria utilizo un flag ('firstPt'), que se
                    pone a 'False' tras haberse pasado por primera vez por 'self.splitSegment'.
                    '''
                    self.splitSegment(trajectories[key][0][contSeg], trajectories[key][1][contSeg], key, firstPt)
                    firstPt = False
        if verbose == True:
            plt.figure(figsize = (15, 7))
            plt.suptitle('flow\ndensity = '+str(self.density)+' | fill = '+str(self.fill)+' | subSampleWinSizePx = '+str(self.subSampleWinSizePx)+'[px]')
            plt.subplot(121)
            plt.title('streamplot')
            plt.streamplot(motVect.x[0, :], motVect.img2DCurr.shape[0] - 1 - motVect.y[:, 0], self.u, -self.v, density = self.density, color = self.modules, linewidth = lw, cmap = 'YlOrRd')
            drawVectorFieldOnImage(motVect.img2DCurr, motVect.x, motVect.y, self.u, self.v, intervalSec = motVect.intervalSec)           
            plt.subplot(122)
            plt.title('stream points trajectories points')
            '''
            Al igual que en el anterior subplot, dibujo la imagen con los vectores de movimiento.
            '''
            drawVectorFieldOnImage(motVect.img2DCurr, motVect.x, motVect.y, self.u, self.v, intervalSec = motVect.intervalSec)
            '''
            Una vez dibujada la imagen con los vectores de movimiento dibujo las trayectorias.
            '''
            self.drawStreamsOnImage(motVect.img2DCurr, palette = 'yellow-red')
            plt.show()
     
    '''
    El metodo 'fillNullValues' hace que se rellenen todos aquellos
    valores invalidos de 'u'/'v' con el valor mas cercano.
    '''
    def fillNullValues(self, motVectObj):
        u = motVectObj.u.copy()
        v = motVectObj.v.copy()
        '''
        Como 'motVectObj' es un objeto de vectores de movimiento, se asume que
        el valor nulo es marcado con np.nan y que los valores nulos se hallan
        en las mismas posiciones de 'u' y 'v'.
        '''
        maskValue = np.isnan(u) & np.isnan(v)
        fil, col = maskValue.shape
        while True in maskValue:
            for f in range(fil):
                for c in range(col):
                    if maskValue[f, c] == True:
                        filPrev, filNext, colPrev, colNext = getKernelEdges(f, c, (fil, col), 3)
                        if False in maskValue[filPrev: (filNext + 1), colPrev: (colNext + 1)]:
                            u[f, c] = u[filPrev: (filNext + 1), colPrev: (colNext + 1)][np.isnan(u[filPrev: (filNext + 1), colPrev: (colNext + 1)]) == False].mean()
                            v[f, c] = v[filPrev: (filNext + 1), colPrev: (colNext + 1)][np.isnan(v[filPrev: (filNext + 1), colPrev: (colNext + 1)]) == False].mean()
                            maskValue[f, c] = False
        return u, v
    
    def getWidthFromSpeeds(self, speeds):
        '''
        Para evitar warnings en 'dataUtils.changeRange' me aseguro de que 'speeds'
        contenga valores diferentes.
        '''
        if speeds.min() == speeds.max():
            return np.zeros_like(speeds) + ((0.05 + 1)/2.)
        else:
            return dataUtils.changeRange(speeds, speeds.min(), speeds.max(), 0.05, 1)
    
    def getSpeedFromWidth(self, width):
        return (width * self.modules.max()) / float(max(self.lineWidths))
    
    def getColorFromSpeed(self, speeds, palette = 'yellow-red'):
        colors = np.zeros((len(speeds), 4), dtype = 'float')
        colors[:, -1] = 1.
        if palette == 'yellow-red':
            '''
            El valor 1 de la componente G dara como resultado un color amarillo, propuesto como valor minimo.
            El valor 0 de la componente G dara como resultado un color rojo, propuesto como valor maximo.
            '''
            colors[:, 0] = 1.
            '''
            Para evitar warnings en 'dataUtils.changeRange' me aseguro de que 'speeds'
            contenga valores diferentes.
            '''
            if speeds.min() == speeds.max():
                colors[:, 1] = (1 - (np.zeros_like(speeds) + ((0.0 + 1.0) / 2.)))
            else:
                colors[:, 1] = (1 - dataUtils.changeRange(speeds, speeds.min(), speeds.max(), 0.0, 1.0))
        elif palette == 'gray-black':
            '''
            Para evitar warnings en 'dataUtils.changeRange' me aseguro de que 'speeds'
            contenga valores diferentes.
            '''
            if speeds.min() == speeds.max():
                grayValue = 1 - (np.zeros_like(speeds) + ((0.5 + 1.0) / 2.))
            else:
                grayValue =  1 - dataUtils.changeRange(speeds, speeds.min(), speeds.max(), 0.5, 1.0)
            colors[:, 0] = grayValue
            colors[:, 1] = grayValue
            colors[:, 2] = grayValue
        elif palette == 'blues':
            '''
            Para evitar warnings en 'dataUtils.changeRange' me aseguro de que 'speeds'
            contenga valores diferentes.
            '''
            if speeds.min() == speeds.max():
                colors[:, 2] = 1 - (np.zeros_like(speeds) + ((0.01 + 0.7) / 2.))
            else:
                colors[:, 2] = 1 - dataUtils.changeRange(speeds, speeds.min(), speeds.max(), 0.01, 0.7)
        else:
            return  
        return colors
    
    def addPt2TrajsPts(self, key, x, y, speed):
        if key in self.trajsPts.keys():
            self.trajsPts[key] = np.vstack((self.trajsPts[key], np.array([x, y, speed])))
        else:
            self.trajsPts[key] = np.array([[x, y, speed]])
    
    def splitSegment(self, segmPts, width, key, firstPt):
        pt0 = (segmPts[0, 0], segmPts[0, 1])
        pt = (segmPts[1, 0], segmPts[1, 1])
        slope = dataUtils.getSlope(pt0, pt)
        '''
        El primer punto de un segmento se guarda si el segmento es el primero
        de la trayectoria. De lo contrario no es que no se almacene, sino que
        para evitar puntos duplicados, se almacena siempre el ultimo punto de
        cada segmento.
        '''
        if firstPt == True:
            self.addPt2TrajsPts(key, pt0[0], pt0[1], self.getSpeedFromWidth(width))
        if dataUtils.isSlopeHorizontal(slope):
            '''
            Si el segmento esta mas definido en el eje 'x' fijo valores de 'x'
            para obtener valores de 'y'.
            '''
            if pt[0] > pt0[0]:
                iterX = np.arange((pt0[0] + funcDefStep), pt[0], funcDefStep)
            else:
                iterX = np.arange((pt0[0] - funcDefStep), pt[0], -funcDefStep)
            for xFixed in iterX:
                yCalc = dataUtils.getY(slope, pt0, xFixed)
                self.addPt2TrajsPts(key, xFixed, yCalc, self.getSpeedFromWidth(width))
        else:
            '''
            Si el incremento es mas grande en el eje 'y' propongo
            valores de 'y' para obtener valores de 'x'.
            x = ((y - y0) / slope) + x0
            '''
            if pt[1] > pt0[1]:
                iterY = np.arange((pt0[1] + funcDefStep), pt[1], funcDefStep) 
            else:
                iterY = np.arange((pt0[1] - funcDefStep), pt[1], -funcDefStep)
            for yFixed in iterY:
                xCalc = dataUtils.getX(slope, pt0, yFixed)
                self.addPt2TrajsPts(key, xCalc, yFixed, self.getSpeedFromWidth(width))
        '''
        El ultimo punto del segmento siempre se guarda.
        '''
        self.addPt2TrajsPts(key, pt[0], pt[1], self.getSpeedFromWidth(width))
    
    def drawSingleStreamOnImg(self, stream, img, palette = None):
        '''
        Para cada trayectoria calculo el grosor de cada punto y el tono de color
        en base a su velocidad.
        '''
        colorDraw = self.getColorFromSpeed(stream[:, 2], palette = palette)
        plt.scatter(stream[:, 0], stream[:, 1], color = colorDraw, s = self.getWidthFromSpeeds(stream[:, 2]))
        plt.arrow(stream[(len(stream)/2), 0], stream[(len(stream)/2), 1], (stream[(len(stream)/2) + 1, 0] - stream[(len(stream)/2), 0]), (stream[(len(stream)/2) + 1, 1] - stream[(len(stream)/2), 1]), color = colorDraw[(len(stream)/2)], head_width = 0.02 * min(img.shape[0], img.shape[1]))
    
    def drawStreamsOnImage(self, img, palette = None):
        for key in self.trajsPts.keys():
            self.drawSingleStreamOnImg(self.trajsPts[key], img, palette = palette)
#             plt.annotate(str(key), (self.trajsPts[key][(len(self.trajsPts[key])/2), 0], self.trajsPts[key][(len(self.trajsPts[key])/2), 1]), (self.trajsPts[key][(len(self.trajsPts[key])/2), 0], self.trajsPts[key][(len(self.trajsPts[key])/2), 1]), color = 'red')
    
    '''
    El metodo 'displaceTraj' devuelve una trayectoria que pasa por
    el punto requerido 'ptReq' y la consigue desplazando la trayectoria
    mas cercana, representada por el registro 'nearestReg' hacia el
    punto requerido siguiendo la direccion ptNearest - ptReq.
    '''
    def getDisplacedTraj(self, ptReq, nearestReg, traj):
        dispTraj = traj.copy()
        diffX = ptReq[0] - nearestReg[1] 
        diffY = ptReq[1] - nearestReg[2] 
        dispTraj[:, 0] = dispTraj[:, 0] + diffX
        dispTraj[:, 1] = dispTraj[:, 1] + diffY
        return dispTraj
    
    '''
    El metodo 'getNearestStreamLine' devuelve la trayectoria mas
    cercana a un punto dado solo que desplazada para hacer que pase
    por el punto deseado.
    '''
    def getNearestStreamLine(self, ptReq, verboseImg = None):
        allPts = self.mergeAllTrajs()
        regNearest = self.getNearestReg(allPts, ptReq)
        dispTraj = self.getDisplacedTraj(ptReq, regNearest, self.trajsPts[regNearest[0]])
        if verboseImg is not None:
            plt.figure(figsize = (15, 10))
            plt.suptitle('nearest trajectory displacement')
            plt.imshow(verboseImg, interpolation = 'none', cmap = 'gray')
            plt.axis('off')
            self.drawStreamsOnImage(verboseImg, palette = 'gray-black')
            self.drawSingleStreamOnImg(dispTraj, verboseImg, palette = 'blues')
            self.drawSingleStreamOnImg(self.trajsPts[regNearest[0]], verboseImg, palette = 'yellow-red')
            plt.scatter(ptReq[0], ptReq[1], s = 30, c = 'yellow')
            plt.scatter(regNearest[1], regNearest[2], s = 30, c = 'red')
            plt.show()
        return dispTraj
    
    def mergeAllTrajs(self, direction = None):
        allPts = None
        for key in self.trajsPts.keys():
            '''
            Copiare todos los puntos menos 1 de cada la trayectoria,
            que puede ser el primero ("direction == 'backward'")
            o el ultimo ("direction == 'forward'"), ya que de ellos
            no se puede saber el incremento.
            '''
            trajPt = np.zeros(((len(self.trajsPts[key]) - 1), 4))
            if direction == 'forward':
                trajPt[:, 1:] = self.trajsPts[key][: -1].copy()
            elif direction == 'backward':
                trajPt[:, 1:] = self.trajsPts[key][1: ].copy()
            else:
                trajPt[:, 1:] = self.trajsPts[key].copy()
            trajPt[:, 0] = key
            if allPts is None:
                allPts = trajPt
            else:
                allPts = np.vstack((allPts, trajPt))
        return allPts
    
    '''
    El metodo 'getSemiRectaPts' obtiene las coordenadas de los puntos
    que conforman una semi recta que parte de 'ptReq' y finaliza en un
    borde de la imagen.
    '''
    def getSemiRectaPts(self, ptReq, pt0, imgShape):
        semiRecta = None
        slope = dataUtils.getSlope(pt0, (ptReq[0], ptReq[1]))
        if dataUtils.isSlopeHorizontal(slope):
            '''
            Si la pendiente se halla entre -1 y 1 (la semi recta esta mejor
            definida en el eje 'x') fijo valores de 'x' para obtener valores
            de 'y'.
            y = (slope * (x - x0)) + y0
            '''
            if ptReq[0] > pt0[0]:
                iterX = np.arange(ptReq[0], imgShape[1] - 1, funcDefStep)
            else:
                iterX = np.arange(ptReq[0], 0, -funcDefStep)
            for xFixed in iterX:
                yCalc = dataUtils.getY(slope, ptReq, xFixed)
                if semiRecta is None:
                    semiRecta = np.array([[xFixed, yCalc]])
                else:
                    semiRecta = np.vstack((semiRecta, np.array([xFixed, yCalc])))
        else:
            '''
            Si la semi recta esta mejor definida en el eje 'y' fijo valores 
            de 'y' para obtener valores de 'x'.
            x = ((y - y0) / slope) + x0
            '''
            if ptReq[1] > pt0[1]:
                iterY = np.arange(ptReq[1], imgShape[0] - 1, funcDefStep) 
            else:
                iterY = np.arange(ptReq[1], 0, -funcDefStep)
            for yFixed in iterY:
                xCalc = dataUtils.getX(slope, ptReq, yFixed)
                if semiRecta is None:
                    semiRecta = np.array([[xCalc, yFixed]])
                else:
                    semiRecta = np.vstack((semiRecta, np.array([xCalc, yFixed])))
        return semiRecta
    
    def getNearestReg(self, allPts, ptReq):
        diffNearTraj = np.sqrt(np.power((allPts[:, 1] - ptReq[0]), 2) + np.power((allPts[:, 2] - ptReq[1]), 2))
        return allPts[diffNearTraj == diffNearTraj.min()][0]
    
    def getRegOtherSide(self, allPts, ptReq, regNear, imgShape):
        '''
        Una vez identificada la trayectoria mas cercana y el punto mas cercano
        a un lado del punto de analisis me dispongo a encontrar la trayectoria y el punto
        mas cercano al otro lado del punto de analisis. Para ello busco en la direccion
        trazada por la recta determinada por los dos puntos que se tienen: el de analisis y el
        mas cercano. 
        '''
        semiRecta = self.getSemiRectaPts(ptReq, (regNear[1], regNear[2]), imgShape)
        '''
        Selecciono aquellos puntos de las trayectorias 'allPts' cuyas coordenadas 'x' e 'y'
        difieran en menos de 'funcDefStep' de los puntos de la semi recta 'semiRecta'.
        '''
        if semiRecta is None:
            return
        intersecciones = None
        for pt in semiRecta:
            '''
            De cada punto de la semi recta miro si la diferencia con los puntos
            de 'allPts' esta dentro del margen xSemiRecta +- funcDefStep
            y ySemiRecta +- funcDefStep
            '''
            posibles = allPts[(allPts[:, 1] > (pt[0] - funcDefStep)) & (allPts[:, 1] < (pt[0] + funcDefStep)) & (allPts[:, 2] > (pt[1] - funcDefStep)) & (allPts[:, 2] < (pt[1] + funcDefStep))] 
            if len(posibles) > 0:
                if intersecciones is None:
                    intersecciones = posibles
                else:
                    intersecciones = np.vstack((intersecciones, posibles))
        '''
        Una vez que tengo todos los puntos candidatos en 'intersecciones'
        selecciono aquel cuya distancia a 'ptReq[0]', 'ptReq[1]' es menor.
        '''
        if intersecciones is None:
            return
        '''
        La variable 'diffPtReq' debe calcularse con respecto al punto de analisis, no
        con respecto al punto de la semi recta mas cercano, pese a que pueda dar como resultado
        un punto que esta mas alejado de la semi recta (aunque mas cercano al punto de analisis).
            
        Con el objetivo de impedir que la trayectoria promediada se cruce con una trayectoria se
        propone evitar que la trayectoria aqui escogida pernezca a otra trayectoria diferente a
        las mas cercana.
        '''
#         diffPtReq = np.sqrt(np.power((intersecciones[:, 1] - ptReq[0]), 2) + np.power((intersecciones[:, 2] - ptReq[1]), 2))
        interseccionesFilt = intersecciones[intersecciones[:, 0] != regNear[0]].copy()
        if len(interseccionesFilt) > 0:
            diffPtReq = np.sqrt(np.power((interseccionesFilt[:, 1] - ptReq[0]), 2) + np.power((interseccionesFilt[:, 2] - ptReq[1]), 2))
            return interseccionesFilt[diffPtReq == diffPtReq.min()][0]
        else:
            return
    
    def getWeights(self, ptReq, reg1, reg2):
        dist1 = np.sqrt(np.power((reg1[1] - ptReq[0]), 2) + np.power((reg1[2] - ptReq[1]), 2))
        dist2 = np.sqrt(np.power((reg2[1] - ptReq[0]), 2) + np.power((reg2[2] - ptReq[1]), 2))
        return (1 - (dist1 / float(dist1 + dist2))), (1 - (dist2 / float(dist1 + dist2)))
    
#     def getAvgPt(self, ptReq, reg1, reg2):
#         '''
#         Lo primero es calcular el peso que cada trayectoria aporta a la
#         interpolacion, sobre 1.
#         '''
#         weight1, weight2 = self.getWeights(ptReq, reg1, reg2)
#         '''
#         Hago el promediado por componentes.
#         
#         Comienzo por la componente 'x'.
#         '''
#         xAvg = (reg1[1] * weight1) + (reg2[1] * weight2)
#         '''
#         Continuo con la componente 'y'.
#         '''
#         yAvg = (reg1[2] * weight1) + (reg2[2] * weight2)
#         '''
#         Finalizo con la componente de velocidad.
#         '''
#         speedAvg = (reg1[3] * weight1) + (reg2[3] * weight2)
#         return np.array([xAvg, yAvg, speedAvg])
    
    def getIdxInTraj(self, reg):
        return np.where((self.trajsPts[reg[0]][:, 0] == reg[1]) & (self.trajsPts[reg[0]][:, 1] == reg[2]) & (self.trajsPts[reg[0]][:, 2] == reg[3]) == True)[0][0]
    
    '''
    El metodo 'getNextPtBySlopeAvg' obtiene el siguiente punto a evaluar
    a partir de un punto de referencia, la media de las pendientes de cada
    punto de las trayectorias circundantes con sus siguientes puntos y la
    constante 'funcDefStep'.
    '''
    def getNextPtBySlopeAvg(self, pt0, regNear, regNearOtherSide, direction):
        if direction == 'forward':
            nextIdxNear = self.getIdxInTraj(regNear) + 1
        elif direction == 'backward':
            nextIdxNear = self.getIdxInTraj(regNear) - 1
        nextPtRegNear = (self.trajsPts[regNear[0]][nextIdxNear, 0], self.trajsPts[regNear[0]][nextIdxNear, 1])
        slopeRegNear = dataUtils.getSlope((regNear[1], regNear[2]), nextPtRegNear)
        if regNearOtherSide is None:
            '''
            Si no existe trayectoria al otro lado del punto de analisis la pendiente media
            es la que determine la trayectoria mas cercana.
            '''
            meanSlope = slopeRegNear
        else:
            '''
            Si existe trayectoria al otro lado del punto de analisis calculo la pendiente
            promediada usando las pendientes de las trayectorias que envelven al punto de
            analisis y utilizando pesos ponderados en funcion de la distancia de cada una
            de ellas al punto de analisis.
            '''
            if direction == 'forward':
                nextIdxNearOtherSide = self.getIdxInTraj(regNearOtherSide) + 1
            elif direction == 'backward':
                nextIdxNearOtherSide = self.getIdxInTraj(regNearOtherSide) - 1
            nextPtRegNearOtherSide = (self.trajsPts[regNearOtherSide[0]][nextIdxNearOtherSide, 0], self.trajsPts[regNearOtherSide[0]][nextIdxNearOtherSide, 1])
            slopeRegNearOtherSide = dataUtils.getSlope((regNearOtherSide[1], regNearOtherSide[2]), nextPtRegNearOtherSide)
            '''
            La obtencion de la pendiente promediada debe ser calculada con pesos ponderados.
            '''
            weightRegNear, weightRegOtherSide = self.getWeights(pt0, regNear, regNearOtherSide)
            meanSlope = (slopeRegNear * weightRegNear) + (slopeRegNearOtherSide * weightRegOtherSide)
        '''
        Basandome en la pendiente media propongo un nuevo valor de 'x'
        o de 'y' para obtener el restante.
        '''
        if dataUtils.isSlopeHorizontal(meanSlope):
            '''
            Como la pendiente es mas horizontal fijo un valor de 'x'
            para obtener 'y'.
            
            Incremento o decremento pt0 en 'x' de acuerdo al sentido de
            crecimiento del registro mas cercano en este eje.
            '''
            if nextPtRegNear[0] > regNear[1]:
                xNext = pt0[0] + 1
            else:
                xNext = pt0[0] - 1
            yNext = dataUtils.getY(meanSlope, pt0, xNext)
        else:
            '''
            Como la pendiente es mas vertical fijo un valor de 'y'
            para obtener 'x'.
            
            Incremento o decremento pt0 en 'y' de acuerdo al sentido de
            crecimiento del registro mas cercano en este eje.
            '''
            if nextPtRegNear[1] > regNear[2]:
                yNext = pt0[1] + 1
            else:
                yNext = pt0[1] - 1
            xNext = dataUtils.getX(meanSlope, pt0, yNext)
        return (xNext, yNext)
    
    def getAvgSpeed(self, ptAnalysis, regNearest, regNearestOtherSide):
        '''
        Si no existe un registro mas cercano en direccion contraria
        se toma la velocidad del punto de la trayectoria mas cercana.
        '''
        if regNearestOtherSide is None:
            return regNearest[3]
        else:
            '''
            Si tengo las dos trayectorias que rodean al punto de analisis
            calculo la velocidad asociada al punto de analisis con una
            media ponderada cuyos pesos dependen de la cercania de cada
            trayectoria al punto de analisis.
            '''
            weightRegNearest, weightRegNearestOtherSide = self.getWeights(ptAnalysis, regNearest, regNearestOtherSide)
            return (regNearest[3] * weightRegNearest) + (regNearestOtherSide[3] * weightRegNearestOtherSide)
    
    '''
    El metodo 'getRemainderAvgPts' adquiere de forma iterativa
    nuevos puntos resultado de promediar puntos de dos trayectorias.
    El proceso de adquisicion (ya sea hacia adelante o hacia atras)
    tiene los siguientes puntos:
        1. Dado un punto de analisis se obtiene el punto mas cercano (que
        pertenecera a alguna trayectoria), 'ptNear'.
        2. Con el punto inicial y el punto mas cercano (que pertenece a
        una trayectoria) calculo el punto mas cercano en direccion opuesta
        (que pertenecera a alguna trayectoria), 'ptNearOtherSide'.
        3. Si no encuentro otro punto en direccion contraria relleno
        con la trayectoria mas cercana.
        4. Si tengo los dos puntos envolventes al punto inicial calculo
        el punto promediado utilizando como ponderacion la distancia desde
        el punto de analisis a cada punto envolvente 'ptNear' y 'ptNearOtherSide'.
        5. Calculo una recta perpendicular a la recta formada por los puntos
        'ptNear' y 'ptNearOtherSide' y que pase por el punto de analisis.
        6. Me muevo 'funcDefStep' elementos hacia adelante/hacia atras generando
        asi un nuevo punto de analisis y volviendo, por tanto, al punto 1.
    '''
    def getRemainderAvgPts(self, allPts, ptAnalysis, imgShape, direction):
        continueLoop = True
        remainderAvgPts = None
        while continueLoop == True:
            '''
            1. Dado un punto de analisis, 'ptAnalysis', encuentro
            el registro mas cercano.
            '''
            regNearest = self.getNearestReg(allPts, ptAnalysis)
            '''
            2. Una vez conocido el registro mas cercano encuentro el
            registro mas cercano en direccion contraria.
            '''
            regNearestOtherSide = self.getRegOtherSide(allPts, ptAnalysis, regNearest, imgShape)
            '''
            3. Una vez que tengo las dos trayectorias que rodean al punto de analisis unicamente
            queda calcular su velocidad asociada como promedio de las velocidades asociadas
            de los dos puntos de las dos trayectorias que rodean al punto de analisis.
            '''
            avgSpeed = self.getAvgSpeed(ptAnalysis, regNearest, regNearestOtherSide)
            '''
            Anado el punto promedio a la trayectoria.
            '''
            if remainderAvgPts is None:
                remainderAvgPts = np.array([[ptAnalysis[0], ptAnalysis[1], avgSpeed]])
            else:
                remainderAvgPts = np.vstack((remainderAvgPts, np.array([ptAnalysis[0], ptAnalysis[1], avgSpeed])))
            '''
            4. Queda por proponer el siguiente punto de analisis, que se obtiene promediando las pendientes
            que forman los puntos de las trayectorias que envuelven al punto de analisis con respecto a su
            correspondiente siguiente punto.
            '''
            ptAnalysis = self.getNextPtBySlopeAvg(ptAnalysis, regNearest, regNearestOtherSide, direction)
            '''
            Las dos condiciones que hacen finalizar el bucle son: que el siguiente punto de analisis se salga
            de la matriz o que entre en un bucle, a efectos practicos, infinito. Esta ultima opcion se controla
            asegurando que el nuevo punto dista mas de 'funcDefStep' de cualquier de los anteriores.
            '''
            if ptAnalysis[0] < 0 or ptAnalysis[0] > (imgShape[1] - 1) or ptAnalysis[1] < 0 or ptAnalysis[1] > (imgShape[0] - 1):
                continueLoop = False
            if np.sqrt(np.power((remainderAvgPts[:, 0] - ptAnalysis[0]), 2) + np.power((remainderAvgPts[:, 1] - ptAnalysis[1]), 2)).min() < funcDefStep:
                continueLoop = False
        return remainderAvgPts
    
#     def getRemainderAvgPts_OLD(self, allPts, ptAnalysis, imgShape, direction):
#         continueLoop = True
#         remainderAvgPts = None
#         while continueLoop == True:
#             '''
#             1. Dado un punto de analisis, 'ptAnalysis', encuentro
#             el registro mas cercano.
#             '''
#             regNearest = self.getNearestReg(allPts, ptAnalysis)
#             '''
#             2. Una vez conocido el registro mas cercano encuentro el
#             registro mas cercano en direccion contraria.
#             '''
#             regNearestOtherSide = self.getRegOtherSide(allPts, ptAnalysis, regNearest, imgShape)
#             '''
#             3. Si no existe un registro mas cercano en direccion contraria
#             se toma como punto el propuesto para analizar y se propone un
#             nuevo punto de analisis copiando la evolucion de la trayectoria
#             mas cercana.
#             '''
#             if regNearestOtherSide is None:
#                 '''
#                 Anado el punto promedio a la trayectoria.
#                 '''
#                 if remainderAvgPts is None:
#                     remainderAvgPts = np.array([[ptAnalysis[0], ptAnalysis[1], regNearest[3]]])
#                 else:
#                     remainderAvgPts = np.vstack((remainderAvgPts, np.array([ptAnalysis[0], ptAnalysis[1], regNearest[3]])))
#                 '''
#                 Obtengo el siguiente punto de analisis copiando la evolucion de la trayectoria
#                 mas cercana.
#                 '''
#                 idx = self.getIdxInTraj(regNearest)
#                 if direction == 'forward':
#                     incrX = self.trajsPts[regNearest[0]][(idx + 1), 0] - self.trajsPts[regNearest[0]][idx, 0] 
#                     incrY = self.trajsPts[regNearest[0]][(idx + 1), 1] - self.trajsPts[regNearest[0]][idx, 1]
#                 elif direction == 'backward':
#                     incrX = self.trajsPts[regNearest[0]][(idx - 1), 0] - self.trajsPts[regNearest[0]][idx, 0] 
#                     incrY = self.trajsPts[regNearest[0]][(idx - 1), 1] - self.trajsPts[regNearest[0]][idx, 1]
#                 ptAnalysis = ((ptAnalysis[0] + incrX), (ptAnalysis[1] + incrY))
#                 if ptAnalysis[0] < 0 or ptAnalysis[0] > (imgShape[1] - 1) or ptAnalysis[1] < 0 or ptAnalysis[1] > (imgShape[0] - 1):
#                     continueLoop = False
#             else:
#                 '''
#                 4. Calculo el punto promediado entre 'regNearest' y 'regNearestOtherSide'
#                 utilizando como ponderacion la distancia desde cada registro envolvente al
#                 punto de analisis.
#                 '''
#                 avgPt = self.getAvgPt(ptAnalysis, regNearest, regNearestOtherSide)
#                 '''
#                 Con el objetivo de evitar que se anadan nuevos puntos muy parecidos a los
#                 ya contenidos en la trayectora promediada 'remainderAvgPts', si un nuevo
#                 punto dista menos de 'funcDefStep/2.' de cualquier de los ya contenidos el
#                 bucle debe interrumpirse.
#                 '''
#                 if remainderAvgPts is not None:
#                     diff = np.sqrt(np.power((remainderAvgPts[:, 0] - avgPt[0]), 2) + np.power((remainderAvgPts[:, 1] - avgPt[1]), 2))
#                     if diff.min() < funcDefStep/2.:
#                         continueLoop = False
#                 '''
#                 Anado el punto promedio a la trayectoria.
#                 '''
#                 if remainderAvgPts is None:
#                     remainderAvgPts = np.array([avgPt])
#                 else:
#                     remainderAvgPts = np.vstack((remainderAvgPts, avgPt))
#                 '''
#                 5. Calculo la direccion del nuevo punto de analisis utilizando como punto de partida
#                 el punto recientemente calculado 'avgPt' y como pendiente la bisectriz del angulo
#                 que se forma por los puntos siguientes a los que indican 'regNearest' y 'regNearestOtherSide'.
#                 6. Obtengo el siguiente punto de analisis desplazandome 'funcDefStep' hacia
#                 adelante en la direccion perpendicular.
#                 '''
#                 ptAnalysis = self.getNextPtBySlopeAvg((avgPt[0], avgPt[1]), regNearest, regNearestOtherSide, direction)
#                 if ptAnalysis[0] < 0 or ptAnalysis[0] > (imgShape[1] - 1) or ptAnalysis[1] < 0 or ptAnalysis[1] > (imgShape[0] - 1):
#                     continueLoop = False
#         return remainderAvgPts
    
    '''
    El metodo 'getAvgStreamLine' devuelve la linea de corriente que
    corresponde a un punto identificado por sus coordenadas 'x'
    e 'y'.
    '''
    def getAvgStreamLine(self, ptIni, imgShape, verboseImg = None):
        '''
        Descompongo la composicion de la trayectoria promediada en
        dos partes: hacia adelante y hacia atras.
        
        En cada bucle de adquisicion de nuevos puntos promediados
        la primera iteracion corresponde al punto de analisis,
        por lo que este punto se hallara en las dos partes adquiridas.
        Comienzo adquiriendo los puntos avanzando en los indices hacia
        adelante.
        '''
        direction  = 'forward' 
        avgTrajForw = self.getRemainderAvgPts(self.mergeAllTrajs(direction), ptIni, imgShape, direction)
        '''
        Ahora me dispongo a realizar el mismo procedimiento pero avanzando
        hacia atras en los indices.
        '''
        direction  = 'backward'
        avgTrajBack = self.getRemainderAvgPts(self.mergeAllTrajs(direction), ptIni, imgShape, direction)
        '''
        La composicion hacia atras es la que incluye a 'ptIni'.
        '''
        avgTraj = np.vstack((np.flipud(avgTrajBack), avgTrajForw[1:]))
        if verboseImg is not None:
            plt.figure(figsize = (15, 10))
            plt.suptitle('averaged stream')
            plt.imshow(verboseImg, interpolation = 'none', cmap = 'gray')
            plt.axis('off')
            self.drawStreamsOnImage(verboseImg, palette = 'gray-black')
            plt.scatter(ptIni[0], ptIni[1], s = 30, c = 'yellow')
            self.drawSingleStreamOnImg(avgTraj, verboseImg, palette = 'blues')
            plt.show()
        return avgTraj
    
#     '''
#     La funcion 'extendTrayectory' extiende una trayectoria hasta que cubra
#     todo el area de la imagen, obtetiendo los puntos 'creados' a traves del
#     ajuste a una funcion polinomial.
#     '''
#     def extendTrayectory(self, traj, img, polyGrade = 1, pts2FuncFitted = 5, verbose = True):
#         fil, col = img.shape
#         extendTraj = None
#         extendTrajMask = None
#         if traj.shape[0] < pts2FuncFitted:
#             pts2FuncFitted = traj.shape[0]
#         '''
#         La manera de proceder es rellenar la primera parte de la trayectoria que
#         no esta definida, en segundo lugar copiar la trayectoria conocida y en
#         tercer lugar rellenar la parte final de la trayectoria.
#         
#         Primero anado la primera parte, inventada.
#         '''
#         if traj[: (pts2FuncFitted  - 1 + 1)][:, 0].max() - traj[: (pts2FuncFitted - 1 + 1)][:, 0].min() > traj[: (pts2FuncFitted - 1 + 1)][:, 1].max() - traj[: (pts2FuncFitted - 1 + 1)][:, 1].min():
#             '''
#             La primera parte de la trayectoria esta mejor definida en el eje 'x'.
#             '''
#             funcBegFitted = np.poly1d(np.polyfit(traj[:(pts2FuncFitted - 1 + 1), 0], traj[:(pts2FuncFitted - 1 + 1), 1], polyGrade))
#             if traj[(pts2FuncFitted - 1), 0] > traj[0, 0]:
#                 rangeXIni = np.arange(0, traj[0, 0], funcDefStep)
#             else:
#                 rangeXIni = np.arange((col-1), traj[0, 0], -funcDefStep)
#             for xProp in rangeXIni:
#                 if funcBegFitted(xProp) >= 0 and funcBegFitted(xProp) <= (fil - 1):
#                     if extendTraj is None:
#                         extendTraj = np.array([[xProp, funcBegFitted(xProp), traj[0, 2]]])
#                         extendTrajMask = np.array([False])
#                     else:
#                         extendTraj = np.vstack((extendTraj, np.array([xProp, funcBegFitted(xProp), traj[0, 2]])))
#                         extendTrajMask = np.vstack((extendTrajMask, False))     
#         else:
#             '''
#             La primera parte de la trayectoria esta mejor definida en el eje 'y'.
#             '''
#             funcBegFitted = np.poly1d(np.polyfit(traj[:(pts2FuncFitted - 1 + 1), 1], traj[:(pts2FuncFitted - 1 + 1), 0], polyGrade))
#             if traj[(pts2FuncFitted - 1), 1] > traj[0, 1]:
#                 rangeYIni = np.arange(0, traj[0, 1], funcDefStep)
#             else:
#                 rangeYIni = np.arange((fil-1), traj[0, 1], -funcDefStep)
#             for yProp in rangeYIni:
#                 if funcBegFitted(yProp) >= 0 and funcBegFitted(yProp) <= (col - 1):
#                     if extendTraj is None:
#                         extendTraj = np.array([[funcBegFitted(yProp), yProp, traj[0, 2]]])
#                         extendTrajMask = np.array([False])
#                     else:
#                         extendTraj = np.vstack((extendTraj, np.array([funcBegFitted(yProp), yProp, traj[0, 2]])))
#                         extendTrajMask = np.vstack((extendTrajMask, False))
#         '''
#         En segundo lugar anado la trayectoria conocida.
#         '''
#         for reg in traj:
#             if extendTraj is None:
#                 extendTraj = reg
#                 extendTrajMask = np.array([True])
#             else:
#                 extendTraj = np.vstack((extendTraj, reg))
#                 extendTrajMask = np.vstack((extendTrajMask, True))
#         '''
#         En tercer lugar anado la parte final, inventada.
#         '''
#         if traj[-pts2FuncFitted:][:, 0].max() - traj[-pts2FuncFitted:][:, 0].min() > traj[-pts2FuncFitted:][:, 1].max() - traj[-pts2FuncFitted:][:, 1].min():
#             '''
#             La ultima parte de la trayectoria esta mejor definida en el eje 'x'.
#             '''
#             funcLastFitted = np.poly1d(np.polyfit(traj[-pts2FuncFitted:, 0], traj[-pts2FuncFitted:, 1], polyGrade))
#             if traj[-1, 0] > traj[-pts2FuncFitted, 0]:
#                 rangeXEnd = np.arange(traj[-1, 0], (col - 1 + 1), funcDefStep)
#             else:
#                 rangeXEnd = np.arange(traj[-1, 0], -1, -funcDefStep)
#             for xProp in rangeXEnd:
#                 if funcLastFitted(xProp) >= 0 and funcLastFitted(xProp) <= (fil - 1):
#                     extendTraj = np.vstack((extendTraj, np.array([xProp, funcLastFitted(xProp), traj[-1, 2]])))
#                     extendTrajMask = np.vstack((extendTrajMask, False))
#         else:
#             '''
#             La ultima parte de la trayectoria esta mejor definida en el eje 'y'.
#             '''
#             funcLastFitted = np.poly1d(np.polyfit(traj[-pts2FuncFitted:, 1], traj[-pts2FuncFitted:, 0], polyGrade))
#             if traj[-1, 1] > traj[-pts2FuncFitted, 1]:
#                 rangeYEnd = np.arange(traj[-1, 1], (fil - 1 + 1), funcDefStep)
#             else:
#                 rangeYEnd = np.arange(traj[-1, 1], -1, -funcDefStep)
#             for yProp in rangeYEnd:
#                 if funcLastFitted(yProp) >= 0 and funcLastFitted(yProp) <= (col - 1):
#                     extendTraj = np.vstack((extendTraj, np.array([funcLastFitted(yProp), yProp, traj[-1, 2]])))
#                     extendTrajMask = np.vstack((extendTrajMask, False)) 
#         if verbose == True:
#             plt.figure(figsize = (15, 10))
#             plt.suptitle('extended trajectory\npolyGrade = '+str(polyGrade)+' | pts2FuncFitted = '+str(pts2FuncFitted))
#             plt.imshow(img, interpolation = 'none', cmap = 'gray')
#             plt.axis('off')
#             self.drawSingleStreamOnImg(extendTraj, img, palette = 'yellow-red')
#             plt.show()
#         return extendTraj, extendTrajMask
    
    def getTimeCross(self, traj, idx0):
        '''
        Ya que la trayectoria se obtiene de los vectores de movimiento,
        se recorre la trayectoria siempre hacia atras para conocer lo que
        vendra.
            
        Cada punto de la trayectoria tiene asociada una velocidad expresada en
        [px/sec]. Si v[px/sec] = e[px] / t[sec]; t[sec] = e[px] / v[px/sec].
        
        Debido a la granularidad de los puntos de la trayectoria, 'n' puntos
        consecutivos pueden tienen la misma velocidad y representar al mismo pixel.
        Cada uno de los 'n' puntos es una unidad.
            
        Cada punto de la trayectoria, por tanto, tiene asociado un tiempo que se
        tarda en recorrer esa unidad. Sin embargo, el espacio no siempre es
        'funcDefStep' puesto que, debido a que los puntos fin de segmento
        siempre se almacenan y unido al hecho de que en ocasiones los segmentos son
        muy pequenos, la distancia entre un punto y otro puede ser menor que
        'funcDefStep', por lo que lo correcto es calcular el espacio como la diferencia
        con respecto al punto anterior, esto es:
            e[px] = float(np.sqrt(np.power((x - x0), 2)) + np.power((y - y0), 2)) 
        Por lo que:
            tCross = float(np.sqrt(np.power((x - x0), 2)) + np.power((y - y0), 2))) / vel
        
        El hecho de que 'idx0' sea un indice me indica que el tamano del
        array que necesito es de un elemento mas.
        
        Creo 'trajMod' para que almacene, por columnas, la coordenada 'x'
        de los puntos, la coordenada 'y', el tiempo que consume en recorrer
        la unidad y, finalmente, el tiempo acumulado desde el inicio del
        recorrido ('idx0').
        '''
        trajMod = np.zeros(((idx0 + 1), 4))
        '''
        Para que se tenga en cuenta la posicion 'idx0' le tengo que sumar '1'
        puesto que es el argumento 'destino'.
        '''
        trajMod[:, 0] = traj[:(idx0 + 1), 0]
        trajMod[:, 1] = traj[:(idx0 + 1), 1]
        trajMod[0, 2] = 0
        trajMod[0, 3] = 0
        '''
        No incluyo el indice 'idx' en el bucle porque se trata de
        la evaluacion, y el tiempo que se tarda en recorrer es 0
        (como se ha indicado en la instruccion 'trajMod[0, 2] = 0' y
        en la instruccion 'trajMod[0, 3] = 0').
        '''
        for contUnit in range((idx0 - 1), -1, -1):
            '''
            Para que el indice 'idx' corresponda a la evaluacion (t0)
            se deben de sumar todos los tiempos exclusivamente anteriores
            al que marque 'contUnit'.
            '''
            espacio  = float(np.sqrt(np.power((trajMod[contUnit, 0] - trajMod[(contUnit + 1), 0]), 2) + np.power((trajMod[contUnit, 1] - trajMod[(contUnit + 1), 1]), 2)))
            trajMod[contUnit, 2] = espacio / float(traj[contUnit, 2])
            trajMod[contUnit, 3] = trajMod[(contUnit + 1):, 2].sum()
        return trajMod
    
    '''
    La funcion 'getForecastPoints' devuelve los puntos de
    una trayectoria por los que se pasara, dado un intervalo de tiempo.
    '''
    def getHorizonPoints(self, ptIni, traj, horRangeSec, verboseImg = None):
        '''
        Para conseguir que punto mas cercano lo mejor es
        hacer el minimo del cuadrado de las diferencias.
        '''
        idx = getPtIndexInTraj(traj, ptIni)
        trajModTimeCross = self.getTimeCross(traj, idx)
        '''
        El primer elemento de prediccion siempre es la evaluacion,
        se indique asi en 'horRangeSec' o no.
        '''
        ptsForecast = np.array([[0, traj[idx, 0], traj[idx, 1]]])
        for tLookFor in horRangeSec:
            '''
            Si el horizonte temporal es '0' (evaluacion) se pasa
            a la siguiente iteracion porque la evaluacion ya se ha
            recogido anteriormente.
            '''
            if tLookFor != 0:
                if tLookFor > trajModTimeCross[:, 3].max():
                    ptsForecast = np.vstack((ptsForecast, np.array([tLookFor, np.nan, np.nan])))
                else:
                    idxT = np.absolute(trajModTimeCross[:, 3] - tLookFor).argmin()
                    ptsForecast = np.vstack((ptsForecast, np.array([tLookFor, traj[idxT, 0], traj[idxT, 1]])))
        if verboseImg is not None:
            plt.figure(figsize = (15, 10))
            plt.suptitle('forecast points')
            plt.imshow(verboseImg, interpolation = 'none', cmap = 'gray')
            plt.axis('off')
            self.drawSingleStreamOnImg(traj, verboseImg, palette = 'yellow-red')
            for reg in ptsForecast:
                plt.scatter(reg[1], reg[2], s = 10, marker = '^', color = 'green')
                plt.annotate('t+'+str(reg[0])+'[sec]', (reg[1], reg[2]), (reg[1], reg[2]), color = 'green')
            plt.show()
        return ptsForecast






















































































#===============================================================================
# CODIGO OBSOLETO/DESCARTADO
#===============================================================================
# '''
# Lo que voy a intentar es hallar correspondencias fiables entre dos imagenes consecutivas.
# Si consiguiese este objetivo, el movimiento experimentado se podria calcular tal
# y como se hace en la clase 'sparseOpticalFlow', a partir de los puntos anteriores y los
# mismos puntos actuales.
# Este modelo no funciona bien.
# Ni siquiera esta terminado porque, visualmente, las correspondencias que encuentra son
# malas e incluso no llega a dar un resultado para 'minSamples = 10'.
# '''
# from skimage.feature import ORB, match_descriptors
# from skimage.transform import AffineTransform
# from skimage.measure import ransac
# class motByCorresp(object):
#     def __init__(self, imgGrayPrev, imgGrayCurr, delaySec, numKeyPts = None, minSamples = 10, residualThr = 3, verbose = False):
#         '''
#         ORB son las siglas de Oriented FAST and Rotated BRIEF. Este algoritmo proviene de 'OpenCV Labs'
#         y se presenta en su paper como una alternativa eficiente a SIFT o SURF en cuanto a coste computacional,
#         comportamiento de las parejas y principalmente las patentes.
#         Para mas informacion sobre ORB visitar el enlace:
#             http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
#         y/o leer los apuntes 'apuntesVisionOrdenador.odt'.
#          
#         La funcion 'ORB' es el detector de caracteristicos y extractor de descriptores binarios FAST orientado y
#         BRIEF rotado.
#         Recibe como parametros:
#             -n_keypoints    int    OPT. Numero de puntos clave a devolver. La funcion devolvera los mejores
#         'n_keypoints' de acuerdo con la respuesta a las esquinas de Harris si mas de 'n_keypoints' son
#         detectados. Si no entonces todos los puntos clave detectados son devueltos.
#             -fast_n    int    OPT. El parametro 'n' en skimage.feature.corner_fast. Minimo numero de pixeles
#         consecutivos fuera de 16 pixeles sobre el circulo que todos deben ser ya sea los pixeles de prueba
#         con respecto al mas brillantes o mas oscuros. Un punto c sobre el circulo es el pixel de prueba con
#         respecto al mas oscuro si Ic < Ip - threshold y el mas brillante si Ic > Ip + threshold. Tambien
#         permanece para el n en el detector de esquinas FAST-n.
#             -fast_threshold    float    OPT. El parametro 'threshold' en feature.corner_fast. El umbral suele
#         decidir cuando los pixeles sobre el circulo son los mas brillantes, los mas oscuros o similares con
#         respecto a los pixeles de prueba. Aminora el umbral cuando se deseen mas esquinas y viceversa.
#             -harris_k    float    OPT. El parametro 'k' en skimage.feature.corner_harris. Factor de sensibilidad
#         para separar esquinas de los bordes. Tipicamente en el rango [0, 0.2]. Valores pequenos de k resulta en
#         la deteccion de esquinas puntiagudas.
#             -downscale    float OPT. Factor de descenso de escala para la piramide de imagen. El valor por
#         defecto 1.2 es escogido asi que hay mas escalas densas que permiten invarianza de escala robusta para
#         una posterior descripcion de caracteristicos.
#             -n_scales    int    OPT. Maximo numero de escalas desde lo mas bajo de la piramide de imagen hasta
#         donde se extraen los caracterisicos.
#         
#         La clase 'ORB' tiene, entre otros atributos:
#             -keypoints    array    ((N, 2) array). Coordenadas de los puntos clave como (filas, columnas).
#             -descriptors    2D array    ((Q, tamano_descriptor) array de tipo buleano). Array bidimensional de
#         descriptores binarios de tamano tamano_descriptor para Q puntos clave despues de apartar los puntos
#         clave en bordes con valores en un indice (i, j) ya sea siendo True o False representando el resultado
#         de la comparacion de intensidad para el i-esimo punto clave sobre la j-esima decision par-pixel. Esto es
#         Q == np.sum(mask).
#         Para mas infrmacion visitar el enlace:
#             http://scikit-image.org/docs/dev/api/skimage.feature.html
#         '''
#         if numKeyPts is None:
#             descExtractPrev = ORB()
#             descExtractCurr = ORB()
#         else:
#             descExtractPrev = ORB(n_keypoints = numKeyPts)
#             descExtractCurr = ORB(n_keypoints = numKeyPts)
#         '''
#         El metodo 'detect_and_extract' de la clase 'ORB' detecta puntos clave orientados FAST y extrae los
#         descriptores rBRIEF. Notese que es mas rapido que primero llamar a la deteccion y despues extraccion.
#         Recibe como parametros:
#             -image    2D array    Array bidimensional.
#         '''
#         descExtractPrev.detect_and_extract(np.uint8(imgGrayPrev))
#         keyPtPrev = descExtractPrev.keypoints
#         descriptPrev = descExtractPrev.descriptors
#         descExtractCurr.detect_and_extract(np.uint8(imgGrayCurr))
#         keyPtCurr = descExtractCurr.keypoints
#         descriptCurr = descExtractCurr.descriptors
#         '''
#         La funcion 'match_descriptors' relaciona por fuerza bruta descriptores.
#         Para cada descriptor en el primer set este relacionador encuentra el descriptor mas cercano en el
#         segundo set (y viceversa en el caso de habilitar 'cross-checking').
#         Recibe como parametros:
#             -descriptors1    array    (M, P) array. Descriptores binarios de tamano P sobre M puntos clave
#         en la primera imagen.
#             -descriptors2    array    (M, P) array. Descriptores binarios de tamano P sobre M puntos clave
#         en la segunda imagen.
#             -metric    string    {'euclidean', 'cityblock', 'minkowsky', 'hamming', ...}. La metrica para
#         calcular la distancia entre dos descriptores. Mira scipy.spatial.distance.cdist para todos los
#         tipos posibles. La distancia de hamming debe ser usada para descriptores binarios. Por defecto la
#         L2-norm es usada para todos los descriptores de tipo float o doble y la distancia de Hamming es
#         usada para descriptores binarios automaticamente.
#             -p    int    La norma-p para aplicar metric = 'minkowski'.
#             -max_distance    float    Maxima distancia permitida entre descriptores de dos puntos clave en
#         imagenes separadas para ser vistas como una relacion.
#             -cross_check    bool    Si es True los puntos clave relacionados son devueltos despues de
#         comprobarlos de forma cruzada. Por ejemplo un par relacionado (keypoint1, keypoint2) es devuelto
#         si keypoint2 es la mejor relacion para keypoint1 en la segunda imagen y keypoint1 es la mejor relacion
#         para keypoint2 en la primera imagen.
#         Devuelve como parametros:
#             -matches    array    (Q, 2) array. Indices de relaciones correspondientes en el primer y segundo
#         set de descriptores, donde matches[:, 0] denota los indices del primero y matches[:, 1] los indices
#         del segundo set de descriptores.
#         
#         Para mas informacion visitar el enlace:
#             http://scikit-image.org/docs/dev/api/skimage.feature.html
#         '''
#         matchesPrev_Curr = match_descriptors(descriptPrev, descriptCurr, metric = 'hamming', cross_check = True)
#         '''
#         Me quedo unicamente con los puntos clave que han pasado la prueba del doble filtro.
#         '''
#         keyPtXFiltPrev = keyPtPrev[matchesPrev_Curr[:, 0]]
#         keyPtXFiltCurr = keyPtCurr[matchesPrev_Curr[:, 1]]
#         '''
#         La funcion 'ransac' ajusta un modelo a unos datos con el algortimo RANSAC (RANdom SAmple Consensus).
#         RANSAC es un algortimo iterativo para la estimacion rubusta de parametros desde un sub-set de inliers
#         desde un set completo de datos. Cada iteracion realiza las siguientes tareas:
#             1- Selecciona 'min_samples' muestras al azar de los datos originales y comprueba si el set de datos
#         es valido (ver 'is_data_valid).
#             2- Estima un modelo para el sub-set aleatorio.
#             3- Clasifica todos los datos como inliners o outliers calculando los residuos a los modelos estimados.
#         Todas las muestras de datos con residuos menores que 'residual_threshold' son considerados como inliers.
#             4- Guarda el modelo estimado como mejor modelo si el numero de muestras inliers es maximo. En el caso
#         de que el modelo estimado actual tenga el mismo numero de inliers, solo es considerado como mejor modelo
#         si tiene una menor suma de residuos.
#         Esos pasos son relizados ya sea un maximo numero de veces o hasta que uno de los criterios especiales de
#         parada sea conocido. El modelo final es estimado usando todas las muestras inliers del previamente
#         determinado mejor modelo.
#         Recibe como parametros:
#             -data    [lista, tupla de] (N, D) array    Set de datos para los que el modelo es ajustado, donde N es
#         el numero de puntos de datos y D la dimensionalidad de los datos. Si la clase del modelo requiere
#         multiples entradas de arrais de datos (por ejemplo origen y destino de coordenadas de
#         skimage.transform.AffineTransform), ellas pueden ser opcionalmente pasadas como tupla o lista. Notese que
#         en este caso las funciones estimate(*data), residuals(*data), is_model_valid(model,*random_data) y
#         is_data_valid(*random_data) deben todas tomar cada array de dato como argumentos separados.
#             -model_class    object    Objeto con los siguientes metodos de objeto:
#                 +success = estimate(*data)
#                 +residuals(*data)
#         donde 'success' indica si la estimacion del modelo es satisfactorio (True o None para exito, False
#         para error).
#             -min_samples    int    Minimo numero de puntos dato para ajustar el modelo.
#             -residual_thresholds    float    Maxima distancia para un punto dato para ser clasificado como inlier.
#             -is_data_valid    funcion    OPT. Esta funcion es llamada con los datos aleatoriamente seleccionados antes
#         de que el modelo sea ajustado a ellos: is_data_valid(*random_data).
#             -is_model_valid    funcion    OPT. Esta funcion es llamada con el modelo estimado y los datos
#         aleatoriamente seleccionados: is_model_valid(model, *random_data).
#             -max_trials    int    OPT. Maximo numero de iteraciones para la seleccion aleatoria de muestras.
#             -stop_sample_num    int    OPT. Detiene las iteraciones si al menos este numero de inliers son hallados.
#             -stop_residuals_sum    float    OPT. Detiene las iteraciones si la suma de los residuos es menor o igual
#         a este umbral.
#             -stop_probability    float en el rango [0, 1]    OPT. Las iteraciones de RANSAC se paran si al menos un
#         set de datos de entrenamiento sin outliers es muestreado con probability >= stop_probability, dependiendo del
#         ratio de inliers del actual mejor modelo y del numero de intentos. Esto requiere generar al menos N muestras
#         (intentos):
#             N >= log(1-probability)/log(1-e**m)
#         donde la probabilidad (confianza) es tipicamente establecida a un valor grande como 0.99 y e es la
#         fraccion actual de inliers con respecto al numero total de muestras.
#         Devuelve como parametros:
#             -model    objeto    Mejor modelo con el mayor set de consenso.
#             -inliers    array    (N, ) array. Mascara buleana de inliers clasificados como True.
#         
#         Para mas informacion visitar el enlace:
#             http://scikit-image.org/docs/dev/api/skimage.measure.html
#         
#         La eleccion del parametro 'min_samples' tendra que ir relacionado con el numero de incognitas que tiene la matriz
#         de trasformacion. Para el caso de la trasformacion afin creo que se necesitan al menos 3 puntos porque se
#         consideran trasformaciones de traslacion (2 incognitas), rotacion (1 incognita), escala (1 incognita), skew
#         (1 incognita), que se podrian conocer con 3 puntos (2 coordenadas por punto aportan 6 variables que son
#         suficientes para obtener las 5 incognitas). Sin embargo, necesito disponer de un numero parecido de puntos de forma
#         que la representacion de este modelo sea igualitaria en comparacion con el numero de muestras que arrojan los
#         otros modelos. Por ello establezco 'minSamples' en 10 muestras.
#         'residual_threshold' se referira a distancia en pixeles (1 - 3 segun videotutorial).
#         '''
#         modelo, inliers = ransac((keyPtXFiltPrev, keyPtXFiltCurr), AffineTransform, min_samples = minSamples, residual_threshold = residualThr)
#         keyPtFinalPrev = keyPtXFiltPrev[inliers == True]
#         keyPtFinalCurr = keyPtXFiltCurr[inliers == True]
#         if verbose:
#             plt.suptitle('delaySec = '+str(delaySec)+'[sec] | numKeyPts = '+str(numKeyPts))
#             plt.subplot2grid((2, 2), (0, 0))
#             plt.title('previous image')
#             plt.imshow(imgGrayPrev, interpolation = 'none', cmap = 'gray')
#             plt.axis('off')
#             nKPt = len(keyPtFinalPrev)
#             for kPt in range(nKPt):
#                 plt.scatter(keyPtFinalPrev[kPt, 1], keyPtFinalPrev[kPt, 0], color = 'red')
#                 plt.annotate(str(kPt), (keyPtFinalPrev[kPt, 1], keyPtFinalPrev[kPt, 0]), (keyPtFinalPrev[kPt, 1], keyPtFinalPrev[kPt, 0]), color = 'red')
#             plt.subplot2grid((2, 2), (1, 0))
#             plt.title('current image')
#             plt.imshow(imgGrayCurr, interpolation = 'none', cmap = 'gray')
#             plt.axis('off')
#             nKPt = len(keyPtFinalCurr)
#             for kPt in range(nKPt):
#                 plt.scatter(keyPtFinalCurr[kPt, 1], keyPtFinalCurr[kPt, 0], color = 'green')
#                 plt.annotate(str(kPt), (keyPtFinalCurr[kPt, 1], keyPtFinalCurr[kPt, 0]), (keyPtFinalCurr[kPt, 1], keyPtFinalCurr[kPt, 0]), color = 'green')
#             plt.show()


# import bob.ip.optflow.hornschunck
# '''
# He comprobado que una imagen procesada consigo misma da como resultado del optical flow
# vectores de modulo 0 como valor maximo y minimo. Visualmente todos parecen tener un
# valor 0 (porque no se ven).
# El optical flow trabaja con valores de luminancia, por eso se le pasan imagenes en
# escala de grises, que es el equivalente a la luminancia de la imagen
# (http://www.uv.es/gpoei/eng/Pfc_web/generalidades/grises/grey.htm).
# El optical flow minimiza una funcion H (funcion de coste) sobre la imagen entera. Asume dos cosas:
#     - La luminancia (brillo) de un pixel no cambia durante el movimiento.
#     - Pixeles vecinos deben representar el mismo movimiento o una parecido.
# Segun lo que yo entiendo, la funcion de coste se escoge de forma que se adapte mejor, en general,
# a toda la imagen. Por esto se dice que se minimiza una funcion, ya que la suma de los errores sera
# la minima posible, que se alcanza mediante un proceso iterativo. De aqui la razon de la existencia
# del parametro 'iterations' que sirve para poner tope a las iteraciones a realizar para ajustar al minimo
# la funcion de coste H.
# La funcion de coste H tiene dos terminos:
#     - H1. Hace referencia a los datos. Variaciones espaciales y temporales de la luminancia. Se detecta
# el movimiento que experimenta un patron de brillo.
#     - H2. Regularizacion (restriccion de la suavidad). Todo viene de utilizar una restriccion anadida.
# Se tiene que pixeles vecinos de un objeto opaco y de tamano finito tienen velocidades similares y
# el campo de velocidades de los patrones de brillo en la imagen varian suavemente en casi cualquier parte.
# De este modo:
#     H = H1 + (alpha*H2)
# donde alpha incrementa o decrementa la influencia de la restriccion de suavidad en el calculo
# del vector. De aqui la existencia del parametro 'alpha'.
# En cuanto al parametro 'iterations', viendo los ejemplos del articulo original, que se puede acceder a traves del enlace:
#     http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=1231385
# , dado que me interesa obtener un vector de movimiento dominante en toda la imagen, adquirire un numero
# de iteraciones de 64, el maximo que se presenta en el articulo.
# En cuanto a la restriccion de suavidad, se utiliza para resolver los problemas de apertura,
# que se relaciona con la imposibilidad de determinar correctamente la direccion de un movimiento viendo tan solo
# una parte de la informacion (que se recoge en la imagen). Este hecho se visualizar muy bien en el
# ejemplo:
#     https://en.wikipedia.org/wiki/Horn%E2%80%93Schunck_method
#     https://en.wikipedia.org/wiki/Motion_perception#/media/File:Aperture_problem_animated.gif
# Aumentar la suavidad puede relacionarse con hacer una funcion mas derivable.
# De momento, ademas de estar seguro de la repercusion del parametro de restriccion de la suavidad
# tuneada con el parametro 'alpha', no sabria medir que valor ponerle de acuerdo al resultado
# que espero obtener. Por ello, de momento, lo dejo con el valor que se sugiere por defecto, 200.
# Tambien me planteo que es mejor, si pasarle imagenes filtradas por la deteccion de nube o sin filtrar.
# Pienso que ya se detecta visualmente un borde en las nubes por lo que prefiero que, 'al otro lado del borde',
# en el cielo, haya valores uniformes con la intencion de que el movimiento detectado en estos puntos donde
# en las dos imagenes hay el mismo valor ('0' con las imagenes filatradas) sea '0'.
# He observado que a medida que aumento el numero de iteraciones el metodo arroja mas resultados distintos
# de '0' alli donde no hay nubes inicial y finalmente. Me planeto lo siguiente:
#     - Si lo que quiero es obtener una corriente de flujo a lo largo de (toda) la imagen utilizare un
# numero de iteraciones mas elevado que me desarrolle el metodo mas veces y me proponga mas resultados
# diferentes de 0.
#     - Si lo que quiero es obtener una solucion que solo tenga en cuenta las regiones de nube aunque la
# solucion alcanzada sea mas pobre, utilizare un numero de iteraciones bajo, 1, en el extremo. La solucion
# es mas pobre porque el error con 1 iteraciones maximo porque no se ha hecho uso del metodo recursivo y
# no se ha minimizado el error. a cambio, solo tengo en cuenta las regiones con nube (puesto que utilizo
# imagenes filtradas). Los vectores que se utilizan con 1 iteracion son todos paralelos entre si.
# He comprobado que, con parametros 'alpha' e 'iterations' iguales, se obtienen resultados diferentes
# segun el tamano de las imagenes que se analicen. Se concluye que el tamano de las imagenes a procesar
# interviene en el resultado.
# La variacion tanto de 'alpha' dejando fijo 'iterations' como al reves los resultados son diferentes.
# 
# En conclusion, mi experiencia con este metodo de calculo de optical flow es el siguiente:
#     - Este metodo me podria facilitar el flujo global del movimiento pero no el modulo del movimiento
# en si.
#     - Me ha sido imposible obtener resultados coherentes incluso para imagenes trucadas y movimientos
# sencillos.
# Finalmente, decido DESCARTAR este metodo.
# '''
# class denseOpticalFlow(object):
#     def __init__(self, imgGrayPrev, imgGrayCurr, alphaParam = 200, iterNum = 64, verbose = False):
#         '''
#         La clase 'opticalFlow' da como resultado del calculo del optical flow arrays del mismo tamano que las
#         imagenes de entrada y consume bastante tiempo. Por ello, por las mismas razones y justificaciones
#         que se dieron en la clase 'openPiv', vamos a reducir el tamano de las imagenes de trabajo.  
#         '''
#         propV = imgWorkSidePx/np.float128(imgGrayPrev.shape[0])
#         propH = imgWorkSidePx/np.float128(imgGrayPrev.shape[1])
#         imgGrayPrevRes = scipy.ndimage.interpolation.zoom(imgGrayPrev, (propH, propV), order=0)
#         imgGrayCurrRes = scipy.ndimage.interpolation.zoom(imgGrayCurr, (propH, propV), order=0)
#         '''
#         La clase 'VanillaFlow' de 'bob.ip.optflow.hornschunck' estima el optical flow entre dos secuencias de
#         de imagenes ('image1', la imagen del comienzo e 'image2', la imagen final). Lo hace usando el metodo iterativo
#         descrito por Horn y Schunck en el paper titulado 'Determining Optical Flow', publicado en 1981, Artificial
#         Intelligence, Vol. 17, No. 1-3, pp. 185-203.
#          
#         Para mas informacion visitar los enlaces:
#             http://pythonhosted.org/bob.ip.optflow.hornschunck/py_api.html?highlight=vanillaflow#bob.ip.optflow.hornschunck.VanillaFlow
#             http://pythonhosted.org/bob.ip.optflow.hornschunck/guide.html
#         Recibe como parametro:
#         (height, width)    tuple    Alto y ancho de las imagenes con las que se va a alimentar el estimador de flujo.
#         '''
#         flow = bob.ip.optflow.hornschunck.VanillaFlow(imgGrayPrevRes.shape)
#         '''
#         El metodo 'estimate' de la clase 'bob.ip.optflow.hornschunck.VanillaFlow' estima el flujo optico que
#         conduce a 'image2'. Este metodo usara la imagen 'conduciente' 'image1' para estimar el flujo optico
#         que conduce a 'image2'. Todas las imagenes de entrada deben ser de arrays bidimensionales de tipo float de 64 bits
#         con la forma (alto, ancho) como se especifica en la construccion del objeto.
#         Recibe como parametros:
#             - alpha    float    factor de poderacion entre la continuidad del brillo y el campo suavizado. En la practica
#         muchos algoritmos consideran valores alrededor de 200 como un buen valor por defecto. A mayor numero mayor
#         importancia se le da al suavizado que estaras poniendo.
#             - iterations    int    numero de iteraciones para minimizar el error del flujo.
#             - image1, image2    array    Array bidimensional de tipo float de 64 bits. Secuencias de imagenes
#         desde las que se estima el flujo.
#             -u, v    array    Array bidimensional de tipo float de 64 bits. Los flujos estimados en las direcciones
#         horizontal y vertical (respectivamente) seran devueltas en estas variables, que deben tener dimensiones que
#         coincidan con aquellas que crean esta funcion. Si no provees arrays para 'u' y 'v', entonces seran asignadas
#         internamente y devueltas. Debes proveer o ninguna 'u' y 'v' o ambas, de lo contrario una excepcion sera
#         alzada. Date cuenta de que, si provees 'u y 'v' que son distintos de cero, seran tomados como valores iniciales para
#         la minimizacion del error. Estos arrays seran actualizados con el valor final del flujo que conduce a
#         'image2'.
#         Devuelve como parametros:
#             -u, v    array    Array bidimensional de tipo float. Flujos estimados en las direcciones
#         horizontal y vertical (respectivamente).
#         Para mas informacion visitar el enlace:
#             http://pythonhosted.org/bob.ip.optflow.hornschunck/py_api.html?highlight=vanillaflow#bob.ip.optflow.hornschunck.VanillaFlow
#         
#         Me he planteado disponer el metodo para que puedan pasarsele como vectores iniciales los resultados del openPiv para que esta
#         informacion sea utilizada como vectores de partida. Sin embargo, posteriormente he rechazado esta idea debido a 2 razones:
#             - Como mi proposito general es comparar diferentes metodos de calculo de movimiento, introduciendo el resultado
#         de otro metodo estoy sesgando los resultados del optical flow.
#             - En una comprobacion rapida me ha parecido ver que cuando le paso vectores iniciales de valor '0' los resultados
#         de los modulos que se arrojan forma siempre el mismo angulo de 45 o -135 grados dependiendo de la direccion del modulo
#         que resulte.
#         Por esta dos razones concluyo no pasarle ningun array de vectores iniciales.
#         '''
#         uTotal, vTotal = flow.estimate(alpha = alphaParam, iterations = iterNum, image1 = np.array(imgGrayPrevRes, dtype = np.float64), image2 = np.array(imgGrayCurrRes, dtype = np.float64))
#         '''
#         Como las funciones implementadas no tienen informacion del tiempo transcurrido, el resultado que ofrezcan sera
#         en unidades de pixeles. Por tanto, divido el resultado entre el tiempo transcurrido (en segundos) para
#         tener el resultado final en pixeles/segundo. Ademas de ello, multiplico por '(1/prop)' en cada eje
#         para poder aplicar los vectores de movimiento a las imagenes de entrada y no a las reducidas imagenes
#         de trabajo. 
#         '''
#         if verbose:
#             plt.suptitle('optical flow from reduced image\nalpha = '+str(round(alphaParam, 3))+' | iterations = '+str(iterNum))
#             plt.subplot2grid((2, 3), (0, 0))
#             plt.title('previous image')
#             plt.imshow(imgGrayPrevRes, interpolation = 'none', cmap = 'gray')
#             plt.axis('off')
#             plt.subplot2grid((2, 3), (1, 0))
#             plt.title('current image')
#             plt.imshow(imgGrayCurrRes, interpolation ='none', cmap = 'gray')
#             plt.axis('off')
#             plt.subplot2grid((2, 3), (0, 1), rowspan = 2, colspan = 2)
#             plt.title('samples of optical flow vectors over current image')
#             plt.imshow(imgGrayCurrRes, interpolation = 'none', cmap = 'gray')
#             plt.axis('off')
#             rows, columns = uTotal.shape
#             for row in range(rows):
#                 for col in range(columns):
#                     if row % densPts == 0 and col % densPts == 0:
#                         plt.arrow(col, row, uTotal[row, col], -vTotal[row, col], width = 0.5, color = 'green')
#             plt.show()