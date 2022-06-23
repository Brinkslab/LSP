# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 17:56:30 2022

@author: mlocarno
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit

path_cs='...'
path_folder='Ellipsoid (1nm mesh size)/'

#Physical constants
c=299792458 #Speed of light [m/s]
vF=1.39e6   #Fermi velocity [m/s]
i=1j        #Imaginary unit

#CVD-friendly qualitatively color cycle
OkabeIto=np.array([[0,0,0],
                   [0.9,0.6,0],
                   [0.35,0.7,0.9],
                   [0,0.6,0.5],
                   [0.95,0.9,0.25],
                   [0,0.45,0.7],
                   [0.8,0.4,0],
                   [0.8,0.6,0.7]])

def fit_dielectric(file='...'):
    """
    This function extracts the data for the dielectric constants of bulk gold and fits them with the corrected Drude model to extract the gold-like model parameters.
    ----------
    file : string, optional
        Location of the .txt file containing the dielectric constants of bulk gold.
    ----------
    Returns the plasma frequency, the damping frequency and the dielectric constant at infinite frequency
    """
    
    #Extract the data
    A=np.loadtxt(file)
    w_data=A[:,0]
    e1=A[:,4]
    e2=A[:,5]
    
    #Plot the data for the real part of epsilon
    fig=plt.subplot(111)
    fig.plot(w_data,e1)
    #Fit the data
    param,_=curve_fit(fit_real,xdata=w_data,ydata=e1,bounds=([1e16,1],[2e16,10]))
    wp_fit=param[0]
    em_inffit=param[1]
    #Plot the fit curve
    fig.plot(np.linspace(9e14,5e15,1000),fit_real(np.linspace(9e14,5e15,1000),wp_fit,em_inffit))
    plt.show()

    #Plot the data for the imaginary part of epsilon   
    fig2=plt.subplot(111)
    fig2.plot(w_data,e2)
    #Fit the data    
    param,_=curve_fit(lambda w,g: fit_imag(w,wp_fit,g),xdata=w_data,ydata=e2,bounds=([1e13],[1e14]))
    g_fit=param[0]
    #Plot the fit curve
    fig2.plot(np.linspace(9e14,5e15,1000),fit_imag(np.linspace(9e14,5e15,1000),wp_fit,g_fit))
    plt.show()
    
    return wp_fit,g_fit,em_inffit

def fit_real(w,wplasma,em_inf):
    """
    Fit function for the real part in the corrected Drude model
    """
    return em_inf-(wplasma**2)/(w**2)

def fit_imag(w,wplasma,g):
    """
    Fit function for the imaginary part in the corrected Drude model
    """
    return (wplasma**2*g)/(w**3+g**2*w)

def calculate_peak(r,eh=1,surfscatter=False,mlwa=False):
    """
    This function calculates analytically the polarizability for a spherical nanoparticle of radius r immersed in a dielectric medium.
    If 'surfscatter' is True, the correction for surface damping is included in the model.
    If 'mlwa' is True, the Modified Long-Wavelength Approximation is included in the model.
    """
    g=gamma
    if surfscatter:
        g=g+3/4*vF/r
    
    #Real part of the dielectric function
    em_r=em_inf-(wp**2)/(w**2+g**2)
    #Imaginary part of the dielectric function
    em_i=(wp**2*g)/(w**3+g**2*w)
    #Complex dielectric function
    em=em_r-em_i*i
    
    #Volume of the nanoparticle
    v=(4/3*np.pi*r**3)
    
    #Polarizability
    a=3*v*(em-eh)/(em+2*eh)
    
    if mlwa:
        MLWA=1/(1-a/(4*np.pi)*((k*np.sqrt(eh))**2/r-i*2/3*(k*np.sqrt(eh))**3))
        a=a*MLWA
    return a/v

def calculate_peak_ellipsoid(ax,ay,az,eh=1,surfscatter=True,mlwa=True):
    """
    This function calculates analytically the polarizability for an ellipsoidal nanoparticle of semiaxes ax, ay,az immersed in a dielectric medium.
    If 'surfscatter' is True, the correction for surface damping is included in the model.
    If 'mlwa' is True, the Modified Long-Wavelength Approximation is included in the model.
    """

    gx=gy=gz=gamma
    if surfscatter:
        gx=gx+3/4*vF/ax
        gy=gy+3/4*vF/ay
        gz=gz+3/4*vF/az
    
    #Calculate the complex dielectric function in the three directions
    em_rx=em_inf-(wp**2)/(w**2+gx**2)
    em_ix=(wp**2*gx)/(w**3+gx**2*w)
    emx=em_rx-em_ix*i
    
    em_ry=em_inf-(wp**2)/(w**2+gy**2)
    em_iy=(wp**2*gy)/(w**3+gy**2*w)
    emy=em_ry-em_iy*i
    
    em_rz=em_inf-(wp**2)/(w**2+gz**2)
    em_iz=(wp**2*gz)/(w**3+gz**2*w)
    emz=em_rz-em_iz*i
    
    #Volume of the nanoparticle
    v=4/3*np.pi*ax*ay*az
    
    #Depolarization factors are not calculated correctly with measures in m, therefore we convert to nm (scale invariance)
    axnm=ax*1e9
    aynm=ay*1e9
    aznm=az*1e9
    Lx=axnm*aynm*aznm/2*quad(lambda q: 1/((q+axnm**2)*np.sqrt((q+axnm**2)*(q+aynm**2)*(q+aznm**2))),0,np.inf)[0]
    Ly=axnm*aynm*aznm/2*quad(lambda q: 1/((q+aynm**2)*np.sqrt((q+axnm**2)*(q+aynm**2)*(q+aznm**2))),0,np.inf)[0]
    Lz=axnm*aynm*aznm/2*quad(lambda q: 1/((q+aznm**2)*np.sqrt((q+axnm**2)*(q+aynm**2)*(q+aznm**2))),0,np.inf)[0]
    
    #Polarizability
    alphax=v*(emx-eh)/(eh+Lx*(emx-eh))
    alphay=v*(emy-eh)/(eh+Ly*(emy-eh))
    alphaz=v*(emz-eh)/(eh+Lz*(emz-eh))
    
    if mlwa:
        MLWAx=1/(1-alphax/(4*np.pi)*((k*np.sqrt(eh))**2/ax-i*2/3*(k*np.sqrt(eh))**3))
        MLWAy=1/(1-alphay/(4*np.pi)*((k*np.sqrt(eh))**2/ay-i*2/3*(k*np.sqrt(eh))**3))
        MLWAz=1/(1-alphaz/(4*np.pi)*((k*np.sqrt(eh))**2/az-i*2/3*(k*np.sqrt(eh))**3))
        alphax=alphax*MLWAx
        alphay=alphay*MLWAy
        alphaz=alphaz*MLWAz
    return alphax/v,alphay/v,alphaz/v

def convert_sphere_to_rod(r,R):
    """
    This function returns the semiaxes value for an ellipsoidal nanoparticle having the same volume of a sphere of radius r, but with aspect ratio ax/ay = ax/az = R
    """
    ax=(R)**(2/3)*r
    ay=(R)**(-1/3)*r
    az=(R)**(-1/3)*r
    
    return ax,ay,az

def cross_sections_sphere(r,alpha,eh):
    """
    This function returns the absorption and scattering cross sections for a sphere of radius r and polarizability alpha, immersed in a medium with dielectric constant eh 
    """
    
    #Volume
    v=4/3*np.pi*r**3
    
    #Calculate polarizability (input value was normalized to the volume)
    alpha=alpha*v   
    
    #Cross section calculation using Rayleigh formulas
    cs_abs=-(k*np.sqrt(eh))*np.imag(alpha)
    cs_sca=((k*np.sqrt(eh))**4/(6*np.pi))*(np.abs(alpha))**2
    return cs_abs, cs_sca

def cross_sections(ax,ay,az,alphax,alphay,alphaz,eh):
    """
    This function returns the absorption and scattering cross sections for a sphere of radius r and polarizability alpha, immersed in a medium with dielectric constant eh 
    """
    
    #Volume
    v=4/3*np.pi*ax*ay*az
    
    #Calculate polarizabilities (input value was normalized to the volume)
    alphax=alphax*v
    alphay=alphay*v
    alphaz=alphaz*v
    
    #Cross section calculation using Rayleigh formulas for each semiaxis
    cs_abs_x=-(k*np.sqrt(eh))*np.imag(alphax)
    cs_abs_y=-(k*np.sqrt(eh))*np.imag(alphay)
    cs_abs_z=-(k*np.sqrt(eh))*np.imag(alphaz)    
    cs_sca_x=((k*np.sqrt(eh))**4/(6*np.pi))*(np.abs(alphax))**2
    cs_sca_y=((k*np.sqrt(eh))**4/(6*np.pi))*(np.abs(alphay))**2
    cs_sca_z=((k*np.sqrt(eh))**4/(6*np.pi))*(np.abs(alphaz))**2
    return cs_abs_x,cs_abs_y,cs_abs_z,cs_sca_x,cs_sca_y,cs_sca_z

def plot_QSA():
    """
    Plot the Quasi-Static Approximation for a gold nanosphere in vacuum and in water (Figure 4)
    """
    global l,k,w
    lmin=400*1e-9
    lmax=600*1e-9
    w=np.linspace(2*np.pi*c/lmax,2*np.pi*c/lmin,int(2*(lmax-lmin)*1e9+1))
    k=w/c
    l=2*np.pi/k
    
    fig= plt.subplot(111)
    fig.set_prop_cycle(color=OkabeIto)
    fig.plot(l*1e9,np.abs(calculate_peak(20e-9,1)),label='\u03B5$_{{h}}$ = 1')
    fig.plot(l*1e9,np.abs(calculate_peak(20e-9,1.77)),label='\u03B5$_{{h}}$ = 1.77')
    fig.set_xlabel('Wavelength [nm]')
    fig.set_ylabel('|\u03B1|/v [a.u.]')
    leg=plt.legend(title='\u03C9$_{{p}}$ = '+'{:.2f}'.format(wp*1e-16)+'  10$^{16}$ rad/s \n\u03B3 = '+'{:.1f}'.format(gamma*1e-13)+'  10$^{13}$ rad/s\n\u03B5$_{{m,\infty}}$ = '+'{:.2f}'.format(em_inf),loc='best')
    leg._legend_box.align = "left"
    plt.savefig('simpeak.png',dpi=300)
    plt.show()    

def plot_surfscatter():
    """
    Plot the surface damping effect for a gold nanosphere in water (Figure 6.a)
    """
    global l,k,w
    lmin=400*1e-9
    lmax=600*1e-9
    w=np.linspace(2*np.pi*c/lmax,2*np.pi*c/lmin,int(2*(lmax-lmin)*1e9+1))
    k=w/c
    l=2*np.pi/k
    
    fig= plt.subplot(111)
    fig.set_prop_cycle(color=OkabeIto)
    fig.plot(l*1e9,np.abs(calculate_peak(100e-9,1.77,surfscatter=True)),label='r = 100 nm')
    fig.plot(l*1e9,np.abs(calculate_peak(40e-9,1.77,surfscatter=True)),label='r = 40 nm')
    fig.plot(l*1e9,np.abs(calculate_peak(20e-9,1.77,surfscatter=True)),label='r = 20 nm')
    fig.plot(l*1e9,np.abs(calculate_peak(10e-9,1.77,surfscatter=True)),label='r = 10 nm')
    fig.plot(l*1e9,np.abs(calculate_peak(5e-9,1.77,surfscatter=True)),label='r = 5 nm')
    fig.set_xlabel('Wavelength [nm]')
    fig.set_ylabel('|\u03B1|/v [a.u.]')
    leg=plt.legend(title='\u03C9$_{{p}}$ = '+'{:.2f}'.format(wp*1e-16)+'  10$^{16}$ rad/s \n\u03B3 = '+'{:.1f}'.format(gamma*1e-13)+'  10$^{13}$ rad/s \n\u03B5$_{{m,\infty}}$ = '+'{:.2f}'.format(em_inf)+'\n\u03B5$_{{h}}$ = 1.77 \nv$_{{F}}$ = 1.39 10$^{6}$ m/s',loc='best')
    leg._legend_box.align = "left"
    plt.savefig('SurfScatter.png',dpi=300)
    plt.show()

def plot_MLWA():
    """
    Plot the Modified Long-Wavelength Approximation effect for a gold nanosphere in water (Figure 6.b)
    """
    global l,k,w
    lmin=400*1e-9
    lmax=800*1e-9
    w=np.linspace(2*np.pi*c/lmax,2*np.pi*c/lmin,int(2*(lmax-lmin)*1e9+1))
    k=w/c
    l=2*np.pi/k
    
    fig= plt.subplot(111)
    fig.set_prop_cycle(color=OkabeIto)
    fig.plot(l*1e9,np.abs(calculate_peak(5e-9,1.77,mlwa=True)),label='r = 5 nm')
    fig.plot(l*1e9,np.abs(calculate_peak(15e-9,1.77,mlwa=True)),label='r = 15 nm')
    fig.plot(l*1e9,np.abs(calculate_peak(20e-9,1.77,mlwa=True)),label='r = 20 nm')
    fig.plot(l*1e9,np.abs(calculate_peak(30e-9,1.77,mlwa=True)),label='r = 30 nm')
    fig.plot(l*1e9,np.abs(calculate_peak(40e-9,1.77,mlwa=True)),label='r = 40 nm')
    fig.set_xlabel('Wavelength [nm]')
    fig.set_ylabel('|\u03B1|/v [a.u.]')
    leg=plt.legend(title='\u03C9$_{{p}}$ = '+'{:.2f}'.format(wp*1e-16)+'  10$^{16}$ rad/s \n\u03B3 = '+'{:.1f}'.format(gamma*1e-13)+'  10$^{13}$ rad/s \n\u03B5$_{{m,\infty}}$ = '+'{:.2f}'.format(em_inf)+'\n\u03B5$_{{h}}$ = 1.77',loc='best')
    leg._legend_box.align = "left"
    plt.savefig('MLWA.png',dpi=300)
    plt.show()

def plot_cross_sections():
    """
    Plot the cross sections for a gold ellipsoid immersed in water, compared to the FDTD simulated results (Figure 7)
    """
    global l,k,w
    lmin=400*1e-9
    lmax=900*1e-9
    w=np.linspace(2*np.pi*c/lmax,2*np.pi*c/lmin,int(2*(lmax-lmin)*1e9+1))
    k=w/c
    l=2*np.pi/k
    
    #Radius of the equivalent nanosphere
    r=25*1e-9
    
    #FDTD data    
    absL_FDTD=np.loadtxt(path_cs+path_folder+'abs_long.txt',delimiter=',',skiprows=3)
    scaL_FDTD=np.loadtxt(path_cs+path_folder+'sca_long.txt',delimiter=',',skiprows=3)
    absL_FDTD[:,0]=absL_FDTD[:,0]*1e9
    scaL_FDTD[:,0]=scaL_FDTD[:,0]*1e9
    absT_FDTD=np.loadtxt(path_cs+path_folder+'abs_trans.txt',delimiter=',',skiprows=3)
    scaT_FDTD=np.loadtxt(path_cs+path_folder+'sca_trans.txt',delimiter=',',skiprows=3)
    absT_FDTD[:,0]=absT_FDTD[:,0]*1e9
    scaT_FDTD[:,0]=scaT_FDTD[:,0]*1e9
    absTOT_FDTD=(absL_FDTD[:,1]+absT_FDTD[:,1])/2
    scaTOT_FDTD=(scaL_FDTD[:,1]+scaT_FDTD[:,1])/2
    extTOT_FDTD=absTOT_FDTD+scaTOT_FDTD
    
    #QSA Sphere model
    alpha=calculate_peak(r,1.77)
    cs_s_abs, cs_s_sca=cross_sections_sphere(r,alpha,1.77)
    cs_s_ext=cs_s_abs+cs_s_sca
    

    #Corrected CM model
    ax,ay,az=convert_sphere_to_rod(r,2)
    alphax,alphay,alphaz=calculate_peak_ellipsoid(ax,ay,az,1.77)
    cs_abs_x=cross_sections(ax,ay,az,alphax,alphay,alphaz,1.77)[0]
    cs_sca_x=cross_sections(ax,ay,az,alphax,alphay,alphaz,1.77)[3]
    cs_abs_y=cross_sections(ax,ay,az,alphax,alphay,alphaz,1.77)[1]
    cs_sca_y=cross_sections(ax,ay,az,alphax,alphay,alphaz,1.77)[4]
    cs_abs=(cs_abs_x+cs_abs_y)/2
    cs_sca=(cs_sca_x+cs_sca_y)/2
    cs_ext=cs_abs+cs_sca


    figa= plt.subplot(111)
    figa.set_prop_cycle(color=OkabeIto)
    figa.plot(absL_FDTD[:,0],absTOT_FDTD/np.max(extTOT_FDTD),label='FDTD')
    figa.plot(l*1e9,cs_s_abs/np.max(cs_s_ext),':',label='Sphere (QSA)')
    figa.plot(l*1e9,cs_abs/np.max(cs_ext),'--',label='Full analytical model')
    figa.set_title('Absorption')
    figa.set_xlabel('Wavelength [nm]')
    figa.set_ylabel('\u03c3$_{abs}$ / max(\u03c3$_{ext}$) [a.u.]')
    leg=plt.legend(loc='best')
    leg._legend_box.align = "left"
    plt.savefig('Compare_Abs.png',dpi=300)
    plt.show()

    figb= plt.subplot(111)
    figb.set_prop_cycle(color=OkabeIto)
    figb.plot(absL_FDTD[:,0],scaTOT_FDTD/np.max(extTOT_FDTD),label='FDTD')
    figb.plot(l*1e9,cs_s_sca/np.max(cs_s_ext),':',label='Sphere (QSA)')
    figb.plot(l*1e9,cs_sca/np.max(cs_ext),'--',label='Full analytical model')
    figb.set_title('Scattering')
    figb.set_xlabel('Wavelength [nm]')
    figb.set_ylabel('\u03c3$_{sca}$ / max(\u03c3$_{ext}$) [a.u.]')
    leg=plt.legend(loc='best')
    leg._legend_box.align = "left"
    plt.savefig('Compare_Sca.png',dpi=300)
    plt.show()

    figc= plt.subplot(111)
    figc.set_prop_cycle(color=OkabeIto)
    figc.plot(absL_FDTD[:,0],extTOT_FDTD/np.max(extTOT_FDTD),label='FDTD')
    figc.plot(l*1e9,cs_s_ext/np.max(cs_s_ext),':',label='Sphere (QSA)')
    figc.plot(l*1e9,cs_ext/np.max(cs_ext),'--',label='Full analytical model')
    figc.set_title('Extinction')
    figc.set_xlabel('Wavelength [nm]')
    figc.set_ylabel('\u03c3$_{ext}$ / max(\u03c3$_{ext}$) [a.u.]')
    leg=plt.legend(loc='best')
    leg._legend_box.align = "left"
    plt.savefig('Compare_Ext.png',dpi=300)
    plt.show()


if __name__=='__main__':
    global wp,gamma,em_inf
    wp,gamma,em_inf=fit_dielectric()

    plot_QSA()
    plot_surfscatter()
    plot_MLWA()
    plot_cross_sections()
