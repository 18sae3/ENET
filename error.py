# import
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt
from numpy import radians as rad

from earth import get_t_earth

from scipy.interpolate import RectBivariateSpline
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, rotate
from scipy.fft import fft, ifft
import tables
import scipy as sp
from numpy import linalg as LA
from skimage.transform import radon, iradon
rEarth = 6371 #km

#differentiates layers
def findLayers(image, bound, theta=0, checkPlot=False, Plot=False): # theta in deg! # plots = false if you just want the layer locations
    image = rotate(image, theta)
    layerz = image[int(image.size**0.5/2),:]

    size = image.size**0.5
    
    xxx = np.linspace(-rEarth*1.2,rEarth*1.2,int(size))
    xv1,yv1 = np.meshgrid(xxx,xxx)

    dx = 2*xxx[-1]/xxx.size
    dydx = np.zeros(xxx.size)
    for i in range(xxx.size-1):
        dydx[i] = (layerz[i+1]-layerz[i])/dx

    
    layerLocs = np.zeros(xxx.size)
    layerDydx = np.zeros(xxx.size)
    for i in range(xxx.size):
        if (abs(dydx[i])>bound and abs(dydx[i+1])<abs(dydx[i]) and abs(dydx[i])>abs(dydx[i-1])):
            layerLocs[i] = xxx[i]
            layerDydx[i] = dydx[i]
    layerLocs = layerLocs[layerLocs!=0]
    layerDydx = layerDydx[layerDydx!=0]

    if checkPlot==True:
        plt.figure()
        plt.plot(xxx,dydx, label='derivative')
        plt.plot(layerLocs,layerDydx, 'o', label='location of layer')
        plt.title("Derivative of Density by Radius at y=0km, 320 Detectors, Boundary= %f" %bound)
        plt.ylabel("change in density (relative)")
        plt.xlabel("radius (km)")
        plt.legend(loc='lower left')
    if Plot==True:
        plt.figure()
        plt.pcolor(xv1,yv1,image)
        plt.plot(layerLocs, [0]*layerLocs.size, 'ro')
        plt.xlabel("(km)")
        plt.ylabel("(km)")
        plt.title("Layer Differentiation for %i Degree Rotation" %(theta))
        plt.colorbar()
    return layerLocs


def AverageLayers(image, bound,numThetas):
    thetas = np.linspace(0,2*np.pi,numThetas)
    numLayers = int(findLayers(image,bound, checkPlot=False, Plot=False).size)
    lyrzAv = np.zeros((numThetas,int(numLayers/2)))
    lyrzAveraged = np.array([])
    for i in range(numThetas):
        lyrz = findLayers(image,bound,thetas[i], checkPlot=False, Plot=False)
        for j in range(int(numLayers/2)):
            lyrzAv[i,j] = abs(lyrz[j]-lyrz[numLayers-j-1])/2
    for i in range(int(numLayers/2)):
        lyrzAveraged = np.append(lyrzAveraged,sum(lyrzAv[:,i])/numThetas)
    # fix the loops and variables they don't make any sense 
    return lyrzAveraged

def error(image, f, step=1):
    err = np.zeros((res,res))
    if f.size != image.size:
        for i in range(0, res, step):
            for j in range(0, res, step):
                if image[i,j] != 0:
                    err[i,j] = abs(f[int(i*f.size**0.5/image.size**0.5),int(j*f.size**0.5/image.size**0.5)]-image[i,j])/image[int(res/2),int(res/2)]
    else:
        for i in range(0, res, step):
            for j in range(0, res, step):
                if image[i,j] != 0:
                    err[i,j] = abs(f[i,j]-image[i,j])/image[int(res/2),int(res/2)]
    err = err[err!=0]
    return sum(err)/err.size
