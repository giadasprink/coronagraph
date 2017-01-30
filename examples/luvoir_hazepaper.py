# Import some standard python packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import pdb
import sys
mpl.rc('font', family='Times New Roman')
mpl.rcParams['font.size'] = 25.0

# Import coronagraph model
import coronagraph as cg

################################
# PARAMETERS
################################

# planet
whichplanet = 'hazyarchean'

# Integration time (hours)
Dt = 10.0
wantsnr = 20.

# Planet params
alpha = 90.     # phase angle at quadrature
Phi   = 1.      # phase function at quadrature (already included in SMART run)
Rp    = 1.0     # Earth radii
r     = 1.0     # semi-major axis (AU)

# Stellar params
Teff  = 5780.   # Sun-like Teff (K)
Rs    = 1.      # star radius in solar radii

# Planetary system params
d    = 3.5     # distance to system (pc)
Nez  = 1.      # number of exo-zodis

# Plot params
plot = True
ref_lam = 0.55
title = ""
ylim =  [-0.1, 0.8]
xlim =  [0, 2.]
tag = ""

# Save params
savefile = False
saveplot = False


################################
# READ-IN DATA
################################

# Read-in spectrum file
if whichplanet == 'earth':
    fn = '../coronagraph/planets/earth_quadrature_radiance_refl.dat'
    model = np.loadtxt(fn, skiprows=8)
    lamhr = model[:,0]
    radhr = model[:,1]
    solhr = model[:,2]

    # Calculate hi-resolution reflectivity
    Ahr   = np.pi*(np.pi*radhr/solhr)

if whichplanet == 'hazyarchean':
    fn = '../coronagraph/planets/Hazy_ArcheanEarth_geo_albedo.txt'
    model = np.loadtxt(fn, skiprows=8)
    lamhr = model[:,0]
    Ahr = model[:,1]
    solhr =  cg.noise_routines.Fstar(lamhr, Teff, Rs, r, AU=True) #sun
    solhrM =  cg.noise_routines.Fstar(lamhr, 3130., 0.3761, 0.12, AU=True) #gj876
    



    
################################
# RUN CORONAGRAPH MODEL
################################

# Run coronagraph with default LUVOIR telescope (aka no keyword arguments)
#lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR = \
#    cg.count_rates(Ahr, lamhr, solhr, alpha, Phi, Rp, Teff, Rs, r, d, Nez\
#                   )

lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR = \
    cg.count_rates(Ahr, lamhr, solhr, alpha,  Rp, Teff, Rs, r, d, Nez, IWA=2., OWA=40., Tsys=260.,\
                   wantsnr=wantsnr, lammin=0.4, THERMAL=True, Res=170., diam=12.7)

lam_M, dlam_M, A_M, q_M, Cratio_M, cp_M, csp_M, cz_M, cez_M, cD_M, cR_M, cth_M, DtSNR_M = \
    cg.count_rates(Ahr, lamhr, solhrM, alpha,  Rp, Teff, Rs, r, d, Nez, IWA=2., OWA=40., Tsys=260.,\
                   wantsnr=wantsnr, lammin=0.4, THERMAL=True, Res=170., diam=12.7)

lam_7, dlam_7, A_7, q_7, Cratio_7, cp_7, csp_7, cz_7, cez_7, cD_7, cR_7, cth_7, DtSNR_7 = \
    cg.count_rates(Ahr, lamhr, solhr, alpha,  Rp, Teff, Rs, r, d, Nez, IWA=2., OWA=40., Tsys=260.,\
                   wantsnr=wantsnr, lammin=0.4, THERMAL=True, Res=170., diam=7.6)

lam_M_7, dlam_M_7, A_M_7, q_M_7, Cratio_M_7, cp_M_7, csp_M_7, cz_M_7, cez_M_7, cD_M_7, cR_M_7, cth_M_7, DtSNR_M_7 = \
    cg.count_rates(Ahr, lamhr, solhrM, alpha,  Rp, Teff, Rs, r, d, Nez, IWA=2., OWA=40., Tsys=260.,\
                   wantsnr=wantsnr, lammin=0.4, THERMAL=True, Res=170., diam=7.6)

lam_5, dlam_5, A_5, q_5, Cratio_5, cp_5, csp_5, cz_5, cez_5, cD_5, cR_5, cth_5, DtSNR_5 = \
    cg.count_rates(Ahr, lamhr, solhr, alpha,  Rp, Teff, Rs, r, d, Nez, IWA=2., OWA=40., Tsys=260.,\
                   wantsnr=wantsnr, lammin=0.4, THERMAL=True, Res=170., diam=5.5)

lam_M_5, dlam_M_5, A_M_5, q_M_5, Cratio_M_5, cp_M_5, csp_M_5, cz_M_5, cez_M_5, cD_M_5, cR_M_5, cth_M_5, DtSNR_M_5 = \
    cg.count_rates(Ahr, lamhr, solhrM, alpha,  Rp, Teff, Rs, r, d, Nez, IWA=2., OWA=40., Tsys=260.,\
                   wantsnr=wantsnr, lammin=0.4, THERMAL=True, Res=170., diam=5.5)

lam_4, dlam_4, A_4, q_4, Cratio_4, cp_4, csp_4, cz_4, cez_4, cD_4, cR_4, cth_4, DtSNR_4 = \
    cg.count_rates(Ahr, lamhr, solhr, alpha,  Rp, Teff, Rs, r, d, Nez, IWA=2., OWA=40., Tsys=260.,\
                   wantsnr=wantsnr, lammin=0.4, THERMAL=True, Res=170., diam=4)

lam_M_4, dlam_M_4, A_M_4, q_M_4, Cratio_M_4, cp_M_4, csp_M_4, cz_M_4, cez_M_4, cD_M_4, cR_M_4, cth_M_4, DtSNR_M_4 = \
    cg.count_rates(Ahr, lamhr, solhrM, alpha,  Rp, Teff, Rs, r, d, Nez, IWA=2., OWA=40., Tsys=260.,\
                   wantsnr=wantsnr, lammin=0.4, THERMAL=True, Res=170., diam=4)


# Calculate background photon count rates
cb = (cz + cez + csp + cD + cR + cth)

# Convert hours to seconds
Dts = Dt * 3600.

# Calculate signal-to-noise assuming background subtraction (the "2")
SNR  = cp*Dts/np.sqrt((cp + 2*cb)*Dts)

# Calculate 1-sigma errors
sig= Cratio/SNR

# Add gaussian noise to flux ratio
spec = Cratio + np.random.randn(len(Cratio))*sig

################################
# PLOTTING
################################

if plot:

    # Create figure
    fig = plt.figure(figsize=(12,10))
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])

    # Set string for plot text
    if Dt > 2.0:
        timestr = "{:.0f}".format(Dt)+' hours'
    else:
        timestr = "{:.0f}".format(Dt*60)+' mins'
    plot_text = r'Distance = '+"{:.1f}".format(d)+' pc'+\
    '\n Integration time = '+timestr

    # If a reference wavelength is specified then return the SNR at that wl
    # corresponding to the integration time given
    if ref_lam:
        ireflam = (np.abs(lam - ref_lam)).argmin()
        ref_SNR = SNR[ireflam]
        plot_text = plot_text + '\n SNR = '+"{:.1f}".format(ref_SNR)+\
            ' at '+"{:.2f}".format(lam[ireflam])+r' $\mu$m'

    # Draw plot
    ax.plot(lam, Cratio*1e9, lw=2.0, color="purple", alpha=0.7, ls="steps-mid")
    ax.errorbar(lam, spec*1e9, yerr=sig*1e9, fmt='o', color='k', ms=5.0)

    # Set labels
    ax.set_ylabel(r"F$_p$/F$_s$ ($\times 10^9$)")
    ax.set_xlabel("Wavelength [$\mu$m]")
    ax.set_title(title)
    ax.text(0.99, 0.94, plot_text,\
         verticalalignment='top', horizontalalignment='right',\
         transform=ax.transAxes,\
            color='black', fontsize=15)

    # Adjust x,y limits
    if ylim is not None: ax.set_ylim(ylim)
    if xlim is not None: ax.set_xlim(xlim)

    # Save plot if requested
    if saveplot:
        plot_tag = "luvoir_demo_"+title+tag+".png"
        fig.savefig(plot_tag)
        print 'Saved: ' + plot_tag
    #else:
    #    plt.show()

#~~~~~~~~~~~~~~~~~~~
    # Create Integration Time figure
    fig = plt.figure(figsize=(11,8))
    #gs = gridspec.GridSpec(1,1)
    #ax = plt.subplot(gs[0])
    #fig.tick_params(axis='both', which='major', labelsize=18)
    #fig.tick_params(axis='both', which='minor', labelsize=10)

   # print lam
  #  print DtSNR

    # Set string for plot text
    if Dt > 2.0:
        timestr = "{:.0f}".format(Dt)+' hours'
    else:
        timestr = "{:.0f}".format(Dt*60)+' mins'
    plot_text = r'Distance = '+"{:.1f}".format(d)+' pc'
    sun_text = 'Sun'
    gj_text = 'GJ 876'
    wantTsnr = "{:.0f}".format(wantsnr)



    # Set labels
#    fig.set_ylabel(r"Integration Time Required for SNR = "+wantTsnr, fontsize=18)
#    fig.set_xlabel("Wavelength [$\mu$m]", fontsize=18)
#    fig.set_title(title)
#    fig.text(0.99, 0.17, plot_text,\
#         verticalalignment='top', horizontalalignment='right',\
#         transform=ax.transAxes,\
#            color='black', fontsize=18)
#    fig.text(0.99, 0.12, sun_text,\
#         verticalalignment='top', horizontalalignment='right',\
#         transform=fig.transAxes,\
#            color='darkturquoise', fontsize=18)
#    fig.text(0.99, 0.07, gj_text,\
 #        verticalalignment='top', horizontalalignment='right',\
 ##        transform=fig.transAxes,\
 #           color='mediumvioletred', fontsize=18)

    sub1 = fig.add_subplot(411)
    # If a reference wavelength is specified then return the SNR at that wl
    # corresponding to the integration time given
    if ref_lam:
        ireflam = (np.abs(lam - ref_lam)).argmin()
        ref_SNR = SNR[ireflam]
        plot_text = plot_text 
    # Draw plot
    sub1.plot(lam, DtSNR, lw=3.0, color="darkturquoise", alpha=1, ls="steps-mid")
    sub1.plot(lam_M, DtSNR_M, lw=3.0, color="mediumvioletred", alpha=1, ls="steps-mid")
    sub1.axis([0.3, 2.1, 0, 300])
    sub1.set_title('15 m segmented')
    #fig.errorbar(lam, spec*1e9, yerr=sig*1e9, fmt='o', color='k', ms=5.0)
            
    sub2 = fig.add_subplot(412)
    # If a reference wavelength is specified then return the SNR at that wl
    # corresponding to the integration time given
    if ref_lam:
        ireflam = (np.abs(lam - ref_lam)).argmin()
        ref_SNR = SNR[ireflam]
        plot_text = plot_text 
    # Draw plot
    sub2.plot(lam, DtSNR, lw=3.0, color="darkturquoise", alpha=1, ls="steps-mid")
    sub2.plot(lam_M, DtSNR_M, lw=3.0, color="mediumvioletred", alpha=1, ls="steps-mid")
    sub2.axis([0.3, 2.1, 0, 300])
    sub2.set_title('9 m segmented')
    #fig.errorbar(lam, spec*1e9, yerr=sig*1e9, fmt='o', color='k', ms=5.0)

    sub3 = fig.add_subplot(413)
    # If a reference wavelength is specified then return the SNR at that wl
    # corresponding to the integration time given
    if ref_lam:
        ireflam = (np.abs(lam - ref_lam)).argmin()
        ref_SNR = SNR[ireflam]
        plot_text = plot_text 
    # Draw plot
    sub3.plot(lam, DtSNR, lw=3.0, color="darkturquoise", alpha=1, ls="steps-mid")
    sub3.plot(lam_M, DtSNR_M, lw=3.0, color="mediumvioletred", alpha=1, ls="steps-mid")
    sub3.axis([0.3, 2.1, 0, 300])
    sub3.set_title('6.5 m segmented')
    #fig.errorbar(lam, spec*1e9, yerr=sig*1e9, fmt='o', color='k', ms=5.0)


    sub4=fig.add_subplot(414)
    # If a reference wavelength is specified then return the SNR at that wl
    # corresponding to the integration time given
    if ref_lam:
        ireflam = (np.abs(lam - ref_lam)).argmin()
        ref_SNR = SNR[ireflam]
        plot_text = plot_text 
    # Draw plot
    sub4.plot(lam, DtSNR, lw=3.0, color="darkturquoise", alpha=1, ls="steps-mid")
    sub4.plot(lam_M, DtSNR_M, lw=3.0, color="mediumvioletred", alpha=1, ls="steps-mid")
    sub4.axis([0.3, 2.1, 0, 300])
    sub4.set_title('4 m monolith')
    #fig.errorbar(lam, spec*1e9, yerr=sig*1e9, fmt='o', color='k', ms=5.0)
                
 
    plt.tight_layout(pad = 0.4)
            
    # Adjust x,y limits
    if ylim is not None: ax.set_ylim([0,300])
    if xlim is not None: ax.set_xlim([0.3,2.1])

    # Save plot if requested
    plot_tag = "luvoir_sun_gj876.eps"
    fig.savefig(plot_tag, format='eps', dpi=1000)
    print 'Saved: ' + plot_tag

    plt.show()

################################
# SAVING
################################

# Save Synthetic data file (wavelength, albedo, error) if requested
if savefile:
    data_tag = 'luvoir_demo_'+tag+'.txt'
    y_sav = np.array([lam,spec,sig])
    np.savetxt(data_tag, y_sav.T)
    print 'Saved: ' + data_tag

sys.exit()
