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
res = 540
n = int(res/1.2)

def nearest(array, value): #finds the index of the nearest value in an array to the given value
        idx = (np.abs(array - value)).argmin()
        return idx

def RFD1(image, m, reconPlots=False,densPlots=False,slicePlots=False,phiError=False, filtering=False): #image = test density model, m = number of detectors
    densities2D = np.zeros((res,m))
    rs = np.linspace(-rEarth,rEarth,res) #evenly spaced parallel "scans"
    thetas = np.linspace(0,2*np.pi,m)
    phis = np.arcsin(rs/rEarth)+np.pi #returns fan beam "scans" that translate to evenly spaced parallel "scans"

    if phiError==True:
        phiErr = 0.573*np.pi/180
        
        for i in range(res):
            phis[i] = phis[i] + 2*phiErr*np.random.random_sample()-phiErr

    thetaPrimes = np.zeros((res,m))
    for i in range(m):
        thetaPrimes[:,i] = ([phis]+thetas[i]-np.pi)%(np.pi*2)
    
    rad = radon(image)
    for i in range(m): #scan across theta primes
        for j in range(res): #scan across phis
            r = rEarth*np.sin(phis[j]-np.pi)
            theta = (thetaPrimes[j,i]+np.pi-phis[j])%(np.pi*2)
            iPrime = nearest(thetas,theta)
            if ((phis[j]>(3*np.pi/2)) or (phis[j]<(np.pi/2))): densities2D[j,iPrime] = 0
            else:
                densities2D[j, iPrime] = rad[nearest(rs,r),int(iPrime*180/m)]

    
    densities2D = densities2D[:, 1:-1]
    
    if densPlots==True:
        plt.figure()
        plt.title("Detected Densities at y=0km")
        plt.plot(xv[int(res/2),:], densities2D[:,int(m/4)]/densities2D[int(m/2),int(m/2)])
        plt.xlabel("radius (km)")
        plt.ylabel("density (relative)")
    
    reconstruction = iradon(densities2D)
    reconstruction = (reconstruction/reconstruction[int(res/2),int(res/2)])
    
    if filtering==True:
        unFilReconstruction = reconstruction
        filtered7 = np.zeros((res,res))
        cutoff = 50
        for i in range(res):
            guy = (fft(reconstruction[i,:]))
            guy[cutoff:] = 0
            filtered7[i,:] = ifft((guy))
        for i in range(res):
            guy = (fft(filtered7[:,i]))
            guy[cutoff:] = 0
            filtered7[:,i] = ifft((guy))
        filtered7 = filtered7-0.185 # resizing done by eye
        filtered7 = filtered7/0.39 # resizing done by eye
        reconstruction = filtered7

    if slicePlots==True:
        plt.figure()
        #plt.plot(xv[int(n/2),:], iradon(rad)[:,int(n/2)], 'b-', label='direct reconstruction')
        plt.plot(xv[int(n/2),:], image[:,int(n/2)], 'r-', label='original image')
        if filtering==True:
            plt.plot(xv[int(n/2),:], unFilReconstruction[:,int(n/2)], 'g-', label='unfiltered reconstruction')
            plt.plot(xv[int(n/2),:], reconstruction[:,int(n/2)], 'b-', label='filtered reconstruction')
        else:
            plt.plot(xv[int(n/2),:], reconstruction[:,int(n/2)], 'g-', label='reconstruction')
            
        plt.title("Reconstructed Densities at y=0km with %i detectors" %m)
        plt.xlabel("radius (km)")
        plt.ylabel("density (relative)")
        plt.legend(loc='lower center')
    if reconPlots==True:
        plt.figure()
        plt.pcolor(xv, yv, (reconstruction))
        plt.colorbar()
        plt.title("2D Reconstruction with %i detectors" %m)
        plt.xlabel("km")
        plt.ylabel("km")
    return reconstruction

def RFD2(image, n, reconPlots=False,densPlots=False,slicePlots=False): #image = test density model, m = number of detectors, n = number of scans
    imagecut = np.zeros((n,n))
    m1 = n
    step = (image.size**0.5/n)
    for i in range(n):
        for j in range(n):
            imagecut[i,j] = image[int(i*step),int(j*step)]
    image = imagecut
    n = int(image.size**0.5)
    m=n
    
    densities2D = np.zeros((n,m))
    rs = np.linspace(-rEarth,rEarth,n) #evenly spaced parallel "scans"
    thetas = np.linspace(0,2*np.pi,m)
    phis = np.arcsin(rs/rEarth)+np.pi #returns fan beam "scans" that translate to evenly spaced parallel "scans"
    _ = np.linspace(-rEarth*1.2,rEarth*1.2,res) # resolution of plot must be equal to number of scans across zenith angle 
    xv, yv = np.meshgrid(_,_)

    thetaPrimes = np.zeros((n,m))
    for i in range(m):
        thetaPrimes[:,i] = ([phis]+thetas[i]-np.pi)%(np.pi*2)
    
    rad = radon(image)
    for i in range(m): #scan across theta primes
        for j in range(n): #scan across phis
            r = rEarth*np.sin(phis[j]-np.pi)
            theta = (thetaPrimes[j,i]+np.pi-phis[j])%(np.pi*2)
            iPrime = nearest(thetas,theta)
            if ((phis[j]>(3*np.pi/2)) or (phis[j]<(np.pi/2))): densities2D[j,iPrime] = 0
            else:
                densities2D[j, iPrime] = rad[nearest(rs,r),int(iPrime*180/n)]
    densities2D = densities2D[:, 1:-1]
    
    if densPlots==True:
        plt.figure()
        plt.title("Detected Densities at y=0km")
        plt.plot(xv[int(n/2),:], densities2D[:,int(m/4)]/densities2D[int(m/2),int(m/2)])
        plt.xlabel("radius (km)")
        plt.ylabel("density (relative)")

    reconstruction = iradon(densities2D)
    reconstruction = (reconstruction/reconstruction[int(n/2),int(n/2)])
    
    if slicePlots==True:
        plt.figure()
        #plt.plot(xv[int(n/2),:], iradon(rad)[:,int(n/2)], 'b-', label='direct reconstruction')
        plt.plot(xv[int(n/2),:], image[:,int(n/2)], 'r-', label='original image')
        plt.plot(xv[int(n/2),:], reconstruction[:,int(n/2)], 'g-', label='reconstruction')
        plt.title("Reconstructed Densities at y=0km with %i detectors" %m1)
        plt.xlabel("radius (km)")
        plt.ylabel("density (relative)")
        plt.legend(loc='lower center')

    if reconPlots==True:
        plt.figure()
        plt.pcolor(xv, yv, (reconstruction))
        plt.colorbar()
        plt.title("2D Reconstruction with %i detectors" %m1)
        plt.xlabel("km")
        plt.ylabel("km")
    return reconstruction
