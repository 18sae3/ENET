import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt

from earth import get_t_earth

import tables
from skimage.transform import radon, iradon
from numpy import linalg as LA


# average radii and densities of earth's layers as given by Perdue University
rEarth = 6371 #km
rUMantle = 6336
rLMantle = 5701
rOuterCore = 3486
rInnerCore = 1216
rhoCrust = 2.0 #g/cm^3
rhoUMantle = 3.3
rhoLMantle = 5.0
rhoOuterCore = 11.0
rhoInnerCore = 13.5

def xvyv(res):
    _ = np.linspace(-rEarth, rEarth, res)
    xv, yv = np.meshgrid(_,_)
    return xv,yv

def nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def drawImage(r,x,y,xElipse,yElipse,rho,n):
    _ = np.linspace(-rEarth, rEarth, n)
    xv, yv = np.meshgrid(_,_)
    image = np.zeros([2,2])
    # Resize Image
    image = np.pad(image, pad_width=int(n/2-1))

    for i in range((np.asarray(r)).size):
        image[(xv/xElipse[i]+x[i])**2+(yv/yElipse[i]+y[i])**2<r[i]**2] = rho[i]
    image = image/rho[-1]
    image = np.pad(image,int(n/10))
    return image

R = [rEarth,rUMantle,rLMantle,rOuterCore,rInnerCore]
X = [0,0,0,0,0]
Y = X
XElipse = [1,1,1,1,1]
YElipse = XElipse
Rho = [rhoCrust,rhoUMantle,rhoLMantle,rhoOuterCore,rhoInnerCore]

def image1(n): 
    return drawImage(R,X,Y,XElipse,YElipse,Rho,n)


def image2(n):
    _ = np.linspace(-rEarth, rEarth, n)
    xv, yv = np.meshgrid(_,_)
    rs = np.linspace(-rEarth,rEarth,n) #evenly spaced parallel "scans"
    thetas = np.linspace(0,np.pi*2,n)
    phis = np.arcsin(rs/rEarth)+np.pi #returns fan beam "scans" that translate to evenly spaced parallel "scans"    

    thetaPrimes = np.zeros((n,n))
    val = 0
    for i in range(n):
        thetaPrimes[:,i] = ([phis]+thetas[i]-np.pi)%(np.pi*2)

    #find column density for desired zenith angle
    densities2D = np.zeros((n,n))
    for i in range(n):
        ts = get_t_earth(phis[i])
        for j in range(n):
            if ((phis[i]>(3*np.pi/2)) or (phis[i]<(np.pi/2))): densities2D[i,j] = 0
            else: 
                jPrime = nearest(thetas,thetaPrimes[i,j])
                densities2D[i,jPrime] = ts #there will be a thetaPrimes[j,i] term with a 2D model
    # the get_t_earth func gets weird at phi=0,2pi
    densities2D[:,0] = 0
    densities2D[:,n-1] = 0 
    #need to get rid of all these zeros! Smooth it somehow!
    numZeros = densities2D[densities2D==0].size
    image = iradon(densities2D)
    image = image/image[int(n/2),int(n/2)]
    image = np.pad(image,int(n/10))
    return image


X3 = [0,0,100,-90,50]
Y3 = [0,0,-100,75,-100]
XElipse3 = [1,1,1.1,0.8,1.1]
YElipse3 = [1,1,0.9,1.1,1]
def image3(n): 
    return drawImage(R,X3,Y3,XElipse3,YElipse3,Rho,n)


def image4(n):
    _ = np.linspace(-rEarth, rEarth, n)
    xv, yv = np.meshgrid(_,_)
    rhoBlob = rhoLMantle*1.5

    image = np.zeros([2,2])
    image = np.pad(image, pad_width=int(n/2-1))

    for i in range((np.asarray(R)).size-2):
        image[(xv/XElipse[i]+X[i])**2+(yv/YElipse[i]+Y[i])**2<R[i]**2] = Rho[i]



    image[(xv-rOuterCore/1.5)**2+(yv-rOuterCore*0.9)**2<(rInnerCore**2/1.75)] = rhoBlob
    image[(xv/.75-rOuterCore*1.25)**2+(yv-rOuterCore*0.8)**2<(rInnerCore**2/1.5)] = rhoBlob
    image[(xv/.75-rOuterCore*1.25)**2+(yv-rOuterCore*0.5)**2<(rInnerCore**2/1.25)] = rhoBlob


    image[(xv+rOuterCore)**2+(yv/0.75+rOuterCore*0.2)**2<(rInnerCore**2/1.5)] = rhoBlob
    image[(xv+rOuterCore*1.1)**2+(yv/0.75+rOuterCore*0.5)**2<(rInnerCore**2/1.5)] = rhoBlob
    image[(xv+rOuterCore*0.9)**2+(yv+rOuterCore*0.5)**2<(rInnerCore**2/1.5)] = rhoBlob


    for i in range((np.asarray(R)).size-2, np.asarray(R).size):
        image[(xv/XElipse[i]+X[i])**2+(yv/YElipse[i]+Y[i])**2<R[i]**2] = Rho[i]


    image = image/Rho[-1]
    image = np.pad(image,int(n/10))
    return image

def image5(n):
    rs = np.linspace(-rEarth,rEarth,n) #evenly spaced parallel "scans"
    phis = np.arcsin(rs/rEarth)+np.pi #returns fan beam "scans" that translate to evenly spaced parallel "scans"
    flavor = 2  # 1,2,3 = e, mu, tau; negative sign for antiparticles
    gamma = 2.  # Power law index of isotropic flux E^-gamma
    ReverseTime = False #You want to go backwards or forward? True for backwards, false for forward in time
    Efinal = 0.5e9 #If you're going backwards in time, set the final neutrino energy. The solution in this case returns a pdf
                # of neutrino energies that would give you Efinal, if propagated forwards. 


    #gamma = 'data/phiHGextrap.dat' #This is an example Honda Gaisser atmospheric flux. You can use this or add your own file, being careful to follow the energy spacing

    def get_att_value(w, v, ci, energy_nodes, zenith, E,phi_in):
        Na = 6.0221415e23
        logE = np.log10(E)
        t = get_t_earth(zenith) * Na
        # g/ cm^2
        #    phi = np.dot(v,(ci*np.exp(w*t)))*energy_nodes**(-2) #this is the attenuated flux
        if(ReverseTime):
            t = -1.*t
            phisol = np.dot(v, (ci * np.exp(w * t))) * energy_nodes**(-2)
            #print phisol
        else:
            phisol = np.dot(v, (ci * np.exp(w * t))) * energy_nodes**(-2) / phi_in #this is phi/phi_inital, i.e. the relative attenuation
        return np.interp(logE, np.log10(energy_nodes), phisol)


    def get_eigs(flavor, gamma, h5_filename, ReverseTime = False, Efinal = None):
        """ Returns the eigenvalues for a given flavor, spectral index, and energy range.

        Args:.
            flavor: specifidies the neutrino flavor of interest. Negative numbers for
                    antineutrinos, positive numbers for neutrinos.
                    1: electron neutrino,
                    2: muon neutrino,
                    and 3: tau neutrino.
            gamma:  If gamma is a string, this is the path and file name of the input spectrum (e.g. an atmospheric flux)
                    If gamma is a number, it is used as the spectral index of the initial flux, E**-gamma.
            h5_filename: complete path and filename of the h5 object that contains the cross sections.
            ReverseTime: Boolean. If True, solves the equations in reverse. i.e propagates the neutrino backwards
            in time. Use with care.
        Returns:
            w: right hand side matrix eigenvalues in unit of cm**2.
            v: right hand side matrix normalized eigenvectors.
            ci: coordinates of the input spectrum in the eigensystem basis.
            energy_nodes: one dimensional numpy array containing the energy nodes in GeV.
            phi_0: E^2 * input spectrum.
        """

        xsh5 = tables.open_file(h5_filename,"r")



        if flavor == -1:
            sigma_array = xsh5.root.total_cross_sections.nuebarxs[:]
        elif flavor == -2:
            sigma_array = xsh5.root.total_cross_sections.numubarxs[:]
        elif flavor == -3:
            sigma_array = xsh5.root.total_cross_sections.nutaubarxs[:]
        elif flavor == 1:
            sigma_array = xsh5.root.total_cross_sections.nuexs[:]
        elif flavor == 2:
            sigma_array = xsh5.root.total_cross_sections.numuxs[:]
        elif flavor == 3:
            sigma_array = xsh5.root.total_cross_sections.nutauxs[:]

        if flavor > 0:
            dxs_array = xsh5.root.differential_cross_sections.dxsnu[:]
        else:
            dxs_array = xsh5.root.differential_cross_sections.dxsnubar[:]

        logemax = np.log10(xsh5.root.total_cross_sections._v_attrs.max_energy)
        logemin = np.log10(xsh5.root.total_cross_sections._v_attrs.min_energy)
        NumNodes = xsh5.root.total_cross_sections._v_attrs.number_energy_nodes
        energy_nodes = np.logspace(logemin, logemax, NumNodes)

        #Note that the solution is scaled by E**2; if you want to modify the incoming
        #spectrum a lot, you'll need to change this here, as well as in the definition of RHS.
        RHSMatrix, sigma_array = get_RHS_matrices(energy_nodes, sigma_array,
                                                dxs_array, ReverseTime)

        #tau regenration
        if flavor == -3:
            RHregen, s1 = get_RHS_matrices(energy_nodes, sigma_array,
                                        xsh5.root.tau_decay_spectrum.tbarfull[:], ReverseTime)
            RHSMatrix = RHSMatrix + RHregen

        elif flavor == 3:
            RHregen, s1 = get_RHS_matrices(energy_nodes, sigma_array,
                                        xsh5.root.tau_decay_spectrum.tfull[:], ReverseTime)
            RHSMatrix = RHSMatrix + RHregen
        elif flavor == -1:
            sigma_array = sigma_array + get_glashow_total(energy_nodes)/2
            RHSMatrix = RHSMatrix + get_glashow_partial(energy_nodes)/2

        # Select initial condition: if gamma is string, load file, otherwise use power law E^-gamma
        if type(gamma) == str:
            phi_in = np.loadtxt(gamma)
            if phi_in.size != energy_nodes.size:
                raise Exception('Input spectrum must have the same size as the energy vector (default 200x1).')
            phi_0 = energy_nodes**2*phi_in
        elif ReverseTime:
            phi_0 = time_reversed_phi0(Efinal, energy_nodes)
        else:
            phi_0 = energy_nodes**(2 - gamma)

        w, v = LA.eig(-np.diag(sigma_array) + RHSMatrix)
        ci = LA.solve(v, phi_0)  # alternative to lstsq solution
        #    ci = LA.lstsq(v,phi_0)[0]

        xsh5.close()

        return w, v, ci, energy_nodes, phi_0


    def get_RHS_matrices(energy_nodes, sigma_array, dxs_array, ReverseTime=False):
        """ Returns the right hand side (RHS) matrices.

        Args:
            energy_nodes: one dimensional numpy array containing the energy nodes in GeV.
            sigma_array: one dimensional numpy array with total cross sections in cm**2.
            dxs_array: two dimensional numpy array with the differential cross section cm**2 GeV**-1.
            ReverseTime: Boolean. If True, solves the equations in reverse. i.e propagates the neutrino backwards
            in time. Use with care.
        Returns:
            RHSMatrix: matrix of size n_nodes*n_nodes containing the E**2 weighted differential
                    cross sections in units of cm**2 GeV.
            sigma_array: one dimensional numpy array containing the total cross section
                        per energy node in cm**2.
        """
        NumNodes = len(energy_nodes)
        DeltaE = np.diff(np.log(energy_nodes))
        RHSMatrix = np.zeros((NumNodes, NumNodes))
        # fill in diagonal terms
        if(ReverseTime):
            for i in range(NumNodes):  #E_out
                for j in range(i+1, NumNodes):  #E_in
                    RHSMatrix[j][i] = DeltaE[j-1] * dxs_array[j][i] * energy_nodes[
                        j]**-1 * energy_nodes[i]**2
            return RHSMatrix, sigma_array
        else:
            for i in range(NumNodes):  #E_out
                for j in range(i + 1, NumNodes):  #E_in
                    # Comparing with NuFate paper: multiply by E_j (= E_in) to account
                    # for log scale, then by E_i^2/E_j^2 to account for variable change
                    # phi -> E^2*phi
                    RHSMatrix[i][j] = DeltaE[j - 1] * dxs_array[j][i] * energy_nodes[
                        j]**-1 * energy_nodes[i]**2
            return RHSMatrix, sigma_array


    #solve the cascade equation once
    w, v, ci, energy_nodes, phi_0 = get_eigs(flavor, gamma, "/home/sofieeklund/nuFATE/resources/NuFATECrossSections.h5", ReverseTime, Efinal)


    E = 100e3  #GeV


    attArr = np.zeros(n)
    for i in range(n):
        attArr[i] = 1/get_att_value(w, v, ci, energy_nodes, phis[i], E,energy_nodes**-gamma)

    attArr2D = np.zeros((n,n))
    for i in range(n):
        attArr2D[:,i] = attArr

    image = iradon(attArr2D)
    image = image/image[int(n/2),int(n/2)]
    image = np.pad(image,int(n/10))
    return image
