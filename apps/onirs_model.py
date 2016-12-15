#####BOKEH O/NIRS NOISE MODEL SIMULATOR#####
# This code produces an interactive browser widget that runs
# the coronagraph noise model
#
#
# To run this code on your local machine, type
# bokeh serve --show onirs_model.py
# 
################################################

# Import some standard python packages

import numpy as np
from astropy.io import fits, ascii 
import pdb
import sys
import os 
from astropy.table import Table, Column
import os
from bokeh.io import curdoc
from bokeh.client import push_session

from bokeh.themes import Theme 
import yaml 
from bokeh.plotting import Figure
from bokeh.models import ColumnDataSource, HBox, VBoxForm, HoverTool, Paragraph, Range1d, DataRange1d, Label, DataSource
from bokeh.models.glyphs import Text
from bokeh.layouts import column, row, WidgetBox 
from bokeh.models.widgets import Slider, Panel, Tabs, Div, TextInput, RadioButtonGroup, Select, RadioButtonGroup
from bokeh.io import hplot, vplot, curdoc, output_file, show, vform
from bokeh.models.callbacks import CustomJS
from bokeh.embed import components, autoload_server


import coronagraph as cg  # Import coronagraph model

#allow it to run it from other folders and still know where planet folder is
planetdir = "../coronagraph/planets/" #new path compared to before
relpath = os.path.join(os.path.dirname(__file__), planetdir)

################################
# PARAMETERS
################################

# Integration time (hours)
Dt = 24.0 # - SLIDER

# Telescopes params
diam = 10. # mirror diameter - SLIDER
Res = 150. # vis resolution - SLIDER
Res_UV = 20. # UV resolution - SLIDER
Res_NIR = 100. #NIR resolution - SLIDER
Tsys = 270. # system temperature - SLIDER

#Object parameters
Ro = 1.         # object radius (solar radii) - SLIDER
#d = 1./206265. # 1 AU
d    = 10.      # distance to object (pc)  - SLIDER 

# Instrumental Params
owa = 3000. #OWA scaling factor - SLIDER
iwa = 0.001 #IWA scaling factor - SLIDER
De = 1e-4 # dark current - 
Re = 0.1 # read noise - 
Dtmax = 1.0 # max single exposure time - SLIDER
wantsnr = 10. #for exposure time calculator - SLIDER

# Template
template = ''
global template
global comparison
global Teff
global Ts

################################
# READ-IN DATA

# Read-in Sun spectrum file to start 

fn = 'earth_quadrature_radiance_refl.dat'
fn = os.path.join(relpath, fn)
model = np.loadtxt(fn, skiprows=8)
lamhr = model[:,0]
radhr = model[:,1]
Fohr = model[:,2] * (1.495e11)**2 / (6.95e8)**2 # convert to flux @ stellar surface
print 'Fohr =', Fohr
# Calculate hi-resolution reflectivity
Fohr_bb = cg.noise_routines.Fstar(lamhr, 5777, 1., 1., AU=True) # stellar flux (comparison)
lammin = 0.2
lammax = 3.
planet_label = ['']
#pdb.set_trace()
Fohr_ = Fohr
lamhr_ = lamhr
Ro_ = Ro


################################
# RUN CORONAGRAPH MODEL
################################

# Run coronagraph with default LUVOIR telescope 
lam, dlam, Fo, q, co, cz, cD, cR, cth, DtSNR = \
    cg.count_rates_onirs(Fohr_, lamhr, Ro, d, lammin=lammin, lammax=lammax, Res=Res, Res_UV = Res_UV, Res_NIR = Res_NIR, diam=diam, Tsys=Tsys, IWA=iwa, OWA=owa,De=De, Re=Re, Dtmax=Dtmax, GROUND=False, THERMAL=True,  wantsnr=wantsnr)

print 'co =', co

# Calculate background photon count rates
cb = (cz + cD + cR + cth)
# Convert hours to seconds
Dts = Dt * 3600.
# Calculate signal-to-noise assuming background subtraction (the "2")
SNR  = co*Dts/np.sqrt((co + 2*cb)*Dts)
# Calculate 1-sigma errors
sig= Fo/SNR
# Add gaussian noise to flux ratio
spec = Fo + np.random.randn(len(Fo))*sig #want to plot planet flux, not counts

#update params
lastlam = lam
snr_ymax = np.max(Fo)
yrange=[snr_ymax]
lamC = lastlam * 0.
Foc = Fo * 0.
global lamC

#data
theobject = ColumnDataSource(data=dict(lam=lam, Fo=Fo, spec=spec, downerr=(spec-sig), uperr=(spec+sig), cz=cz*Dts,  cD=cD*Dts, cR=cR*Dts, cth=cth*Dts, co=co*Dts))
expobject = ColumnDataSource(data=dict(lam=lam[np.isfinite(DtSNR)], DtSNR=DtSNR[np.isfinite(DtSNR)])) 
plotyrange = ColumnDataSource(data = dict(yrange=yrange))
compare = ColumnDataSource(data=dict(lam=lamC, Fo=Foc)) 
expcompare = ColumnDataSource(data=dict(lam=lam[np.isfinite(DtSNR)], DtSNR=DtSNR[np.isfinite(DtSNR)]*(-1000000))) #to make it not show up
textlabel = ColumnDataSource(data=dict(label = planet_label))


################################
# BOKEH PLOTTING
################################
#plots spectrum and exposure time
snr_plot = Figure(plot_height=500, plot_width=750, 
                  tools="crosshair,pan,reset,resize,save,box_zoom,wheel_zoom,hover",
                  toolbar_location='right', x_range=[0.2, 3.0], y_range=[min(Fo)*0.9, max(Fo)*1.1])

exp_plot = Figure(plot_height=500, plot_width=750, 
                  tools="crosshair,pan,reset,resize,save,box_zoom,wheel_zoom,hover",
                  toolbar_location='right', x_range=[0.2, 3.0],
                  y_axis_type="log")

snr_plot.background_fill_color = "beige"
snr_plot.background_fill_alpha = 0.5
snr_plot.yaxis.axis_label='Flux [W/m**2/s]' 
snr_plot.xaxis.axis_label='Wavelength [micron]'
snr_plot.title.text = 'Object Spectrum: Sun at 10 pc' #initial spectrum is Sun at 10 pc

exp_plot.background_fill_color = "beige"
exp_plot.background_fill_alpha = 0.5
exp_plot.yaxis.axis_label='Integration time for SNR = 10 [hours]' 
exp_plot.xaxis.axis_label='Wavelength [micron]'
exp_plot.title.text = 'Planet Spectrum: Sun at 10 pc' #initial spectrum is Sun at 10 pc

snr_plot.line('lam','Fo',source=compare,line_width=2.0, color="navy", alpha=0.7)
snr_plot.line('lam','Fo',source=theobject,line_width=2.0, color="darkgreen", alpha=0.7)
snr_plot.circle('lam', 'spec', source=theobject, fill_color='lightgreen', line_color='black', size=8) 
snr_plot.segment('lam', 'downerr', 'lam', 'uperr', source=theobject, line_width=1, line_color='grey', line_alpha=0.5)

exp_plot.line('lam','DtSNR',source=expcompare,line_width=2.0, color="navy", alpha=0.7)
exp_plot.line('lam','DtSNR',source=expobject,line_width=2.0, color="darkgreen", alpha=0.7)

#text on plot
glyph = Text(x=0.25, y=-0.19, text="label", text_font_size='9pt', text_font_style='bold', text_color='blue')
#attempting to outline the text here for ease of visibility... 
glyph2 = Text(x=0.245, y=-0.19, text="label", text_font_size='9pt', text_font_style='bold', text_color='white')
glyph3 = Text(x=0.25, y=-0.195, text="label", text_font_size='9pt', text_font_style='bold', text_color='white')
glyph4 = Text(x=0.25, y=-0.845, text="label", text_font_size='9pt', text_font_style='bold', text_color='white')
glyph5 = Text(x=0.255, y=-0.19, text="label", text_font_size='9pt', text_font_style='bold', text_color='white')
snr_plot.add_glyph(textlabel, glyph2)
snr_plot.add_glyph(textlabel, glyph3)
snr_plot.add_glyph(textlabel, glyph4)
snr_plot.add_glyph(textlabel, glyph5)
snr_plot.add_glyph(textlabel, glyph)

#hovertool
hover = snr_plot.select(dict(type=HoverTool))
hover.tooltips = [
   ('object', '@co{int}'),
   ('zodi', '@cz{int}'),
   ('dark current', '@cD{int}'),
   ('read noise', '@cR{int}'),
   ('thermal', '@cth{int}')
]

ptab1 = Panel(child=snr_plot, title='Spectrum')
ptab2 = Panel(child=exp_plot, title='Exposure Time')
ptabs = Tabs(tabs=[ptab1, ptab2])
show(ptabs)

################################
#  PROGRAMS
################################

def change_filename(attrname, old, new): 
   format_button_group.active = None 


instruction0 = Div(text="""Specify a filename here:
                           (no special characters):""", width=300, height=15)
text_input = TextInput(value="filename", title=" ", width=100)
instruction1 = Div(text="""Then choose a file format here:""", width=300, height=15)
format_button_group = RadioButtonGroup(labels=["txt", "fits"])
instruction2 = Div(text="""The link to download your file will appear here:""", width=300, height=15)
link_box  = Div(text=""" """, width=300, height=15)


def i_clicked_a_button(new): 
    filename=text_input.value + {0:'.txt', 1:'.fits'}[format_button_group.active]
    print "Your format is   ", format_button_group.active, {0:'txt', 1:'fits'}[format_button_group.active] 
    print "Your filename is: ", filename 
    fileformat={0:'txt', 1:'fits'}[format_button_group.active]
    link_box.text = """Working""" 
 
    t = Table(planet.data)
    t = t['lam', 'spec','cratio','uperr','downerr'] 

    if (format_button_group.active == 1): t.write(filename, overwrite=True) 
    if (format_button_group.active == 0): ascii.write(t, filename)
 
    os.system('gzip -f ' +filename) 
    os.system('cp -rp '+filename+'.gz /home/jtastro/jt-astro.science/outputs') 
    print    """Your file is <a href='http://jt-astro.science/outputs/"""+filename+""".gz'>"""+filename+""".gz</a>. """

    link_box.text = """Your file is <a href='http://jt-astro.science/outputs/"""+filename+""".gz'>"""+filename+""".gz</a>. """


#########################################
# GET DATA FROM USER AND UPDATE PLOT
#########################################

def update_data(attrname, old, new):
    print 'Updating model for exptime = ', exptime.value, ' for planet with R = ', radius.value, ' at distance ', distance.value, ' parsec '
    print '                   exozodi = ', exozodi.value, 'diameter (m) = ', diameter.value, 'resolution = ', resolution.value, 'resolution uv =', resolution_UV.value, 'resolution nir =', resolution_NIR.value,
    print '                   temperature (K) = ', temperature.value, 'IWA = ', inner.value, 'OWA = ', outer.value
    print 'You have chosen planet spectrum: ', template.value
    print 'You have chosen comparison spectrum: ', comparison.value
    try:
       lasttemplate
    except NameError:
       lasttemplate = 'Earth' #default first spectrum
    try:
       lastcomparison
    except NameError:
       lastcomparison = 'none' #default first spectrum
    global lasttemplate
    global Ahr_
    global lamhr_
    global solhr_
    global Teff_
    global Rs_
    global Ahr_c
    global lamhr_c
    global solhr_c
    global Teff_c
    global Rs_c
    global radius_c
    global semimajor_c
    global lastcomparison
    
# Read-in new spectrum file only if changed
    if template.value != lasttemplate:
       if template.value == 'Earth':
          fn = 'earth_quadrature_radiance_refl.dat'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=8)
          lamhr_ = model[:,0]
          radhr_ = model[:,1]
          solhr_ = model[:,2]
          Ahr_   = np.pi*(np.pi*radhr_/solhr_)
          semimajor.value = 1.
          radius.value = 1.
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr, Teff, Rs, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by T. Robinson (Robinson et al. 2011)']


       if template.value == 'Venus':
          fn = 'new_venus.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_ = model[:,0]
          Fhr_ = model[:,3]
          solhr_ = model[:,2]
          Ahr_ = (Fhr_/solhr_) 
          lamhr_ = lamhr_[::-1]
          Ahr_ = Ahr_[::-1]
          semimajor.value = 0.72
          radius.value = 0.94
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by G. Arney']


       if template.value =='Archean Earth':
          fn = 'ArcheanEarth_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=8)
          lamhr_ = model[:,0]
          Ahr_ = model[:,1]
          semimajor.value = 1.
          radius.value = 1.
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by G. Arney (Arney et al. 2016)']
          
       if template.value =='Hazy Archean Earth':
          fn = 'Hazy_ArcheanEarth_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          print fn
          model = np.loadtxt(fn, skiprows=8)
          lamhr_ = model[:,0]
          Ahr_ = model[:,1]
          semimajor.value = 1.
          radius.value = 1.
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by G. Arney (Arney et al. 2016)']


       if template.value =='1% PAL O2 Proterozoic Earth':
          fn = 'proterozoic_hi_o2_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          print fn
          model = np.loadtxt(fn, skiprows=0)
          lamhr_ = model[:,0]
          Ahr_ = model[:,1]
          semimajor.value = 1.
          radius.value = 1.
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by G. Arney (Arney et al. 2016)']
          

       if template.value =='0.1% PAL O2 Proterozoic Earth':
          fn = 'proterozoic_low_o2_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_ = model[:,0]
          Ahr_ = model[:,1]
          semimajor.value = 1.
          radius.value = 1.
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by G. Arney (Arney et al. 2016)']

          
       if template.value =='Early Mars':
          fn = 'EarlyMars_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=8)
          lamhr_ = model[:,0]
          Ahr_ = model[:,1]
          semimajor.value = 1.52
          radius.value = 0.53
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by G. Arney based on Smith et al. 2014']

          
       if template.value =='Mars':
          fn = 'Mars_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=8)
          lamhr_ = model[:,0]
          Ahr_ = model[:,1]
          semimajor.value = 1.52
          radius.value = 0.53         
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by T. Robinson']

          
       if template.value =='Jupiter':
          fn = 'Jupiter_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_ = model[:,0]
          Ahr_ = model[:,1]
          semimajor.value = 5.46
          radius.value = 10.97
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['0.9-0.3 microns observed by Karkoschka et al. (1998); 0.9-2.4 microns observed by Rayner et al. (2009)']

          
       if template.value =='Saturn':
          fn = 'Saturn_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_ = model[:,0]
          Ahr_ = model[:,1]
          semimajor.value = 9.55
          radius.value = 9.14
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['0.9-0.3 microns observed by Karkoschka et al. (1998); 0.9-2.4 microns observed by Rayner et al. (2009)']

          
       if template.value =='Uranus':
          fn = 'Uranus_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_ = model[:,0]
          Ahr_ = model[:,1]
          semimajor.value = 19.21
          radius.value = 3.98
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['0.9-0.3 microns observed by Karkoschka et al. (1998); 0.9-2.4 microns observed by Rayner et al. (2009)']

          
       if template.value =='Neptune':
          fn = 'Neptune_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_ = model[:,0]
          Ahr_ = model[:,1]
          semimajor.value = 29.8
          radius.value = 3.86
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['0.9-0.3 microns observed by Karkoschka et al. (1998); 0.9-2.4 microns observed by Rayner et al. (2009)']

       if template.value =='Warm Neptune at 2 AU':
          fn = 'Reflection_a2_m1.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_ = model[:,0]
          Ahr_ = model[:,1]
          lamhr_ = lamhr_ / 1000. #convert to microns
          Ahr_ = Ahr_ * 0.67 #convert to geometric albedo
          semimajor.value = 2.0
          radius.value = 3.86
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by R. Hu (Hu and Seager 2014)']

       if template.value =='Warm Neptune w/o Clouds at 1 AU':
          fn = 'Reflection_a1_m2.6_LM_NoCloud.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_ = model[:,0]
          Ahr_ = model[:,1]
          lamhr_ = lamhr_ / 1000. #convert to microns
          Ahr_ = Ahr_ * 0.67 #convert to geometric albedo
          semimajor.value = 1.0
          radius.value = 3.86
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by R. Hu (Hu and Seager 2014)']
          
       if template.value =='Warm Neptune w/ Clouds at 1 AU':
          fn = 'Reflection_a1_m2.6_LM.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_ = model[:,0]
          Ahr_ = model[:,1]
          lamhr_ = lamhr_ / 1000. #convert to microns
          Ahr_ = Ahr_ * 0.67 #convert to geometric albedo
          semimajor.value = 1.0
          radius.value = 3.86
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by R. Hu']

       if template.value =='Warm Jupiter at 0.8 AU':
          fn = '0.8AU_3x.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_ = model[:,1]
          Ahr_ = model[:,3]
          semimajor.value = 0.8
          radius.value = 10.97
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by K. Cahoy (Cahoy et al. 2010)']

       if template.value =='Warm Jupiter at 2 AU':
          fn = '2AU_3x.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_ = model[:,1]
          Ahr_ = model[:,3]
          semimajor.value = 2.0
          radius.value = 10.97
          Teff_  = 5780.   # Sun-like Teff (K)
          Rs_    = 1.      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by K. Cahoy (Cahoy et al. 2010)']             
          
       if template.value =='False O2 Planet (F2V star)':
          fn = 'fstarcloudy_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_ = model[:,0]
          Ahr_ = model[:,1]
          semimajor.value = 1.72 #Earth equivalent distance for F star
          radius.value = 1.
          Teff_  = 7050.   # F2V Teff (K)
          Rs_    = 1.3     # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by S. Domagal-Goldman (Domagal-Goldman et al. 2014)']


       if template.value =='Proxima Cen b 10 bar 95% O2 dry':
          fn = 'Proxima15_o2lb_10bar_dry.pt_filtered_hitran2012_50_100000cm_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_ = model[:,0]
          solhr_ = model[:,2]
          Flx_ = model[:,3]
          Ahr_ = Flx_/solhr_
          lamhr_ = lamhr_[::-1]
          Ahr_ = Ahr_[::-1]
          semimajor.value = 0.048
          radius.value = 1.
          distance.value = 1.3
          Teff_  = 3040.   # Sun-like Teff (K)
          Rs_    = 0.141      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by E. Schwieterman (Meadows et al. 2016)']

          
       if template.value =='Proxima Cen b 10 bar 95% O2 wet':
          fn = 'Proxima15_o2lb_10bar_h2o.pt_filtered_hitran2012_50_100000cm_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_ = model[:,0]
          solhr_ = model[:,2]
          Flx_ = model[:,3]
          Ahr_ = Flx_/solhr_
          lamhr_ = lamhr_[::-1]
          Ahr_ = Ahr_[::-1]
          semimajor.value = 0.048
          radius.value = 1.
          distance.value=1.3
          Teff_  = 3040.   # Sun-like Teff (K)
          Rs_    = 0.141      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by E. Schwieterman (Meadows et al. 2016)']

       if template.value =='Proxima Cen b 10 bar O2-CO2':
          fn = 'Proxima16_O2_CO2_10bar_prox_hitran2012_50_100000cm_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_ = model[:,0]
          solhr_ = model[:,2]
          Flx_ = model[:,3]
          Ahr_ = Flx_/solhr_
          lamhr_ = lamhr_[::-1]
          Ahr_ = Ahr_[::-1]
          semimajor.value = 0.048
          radius.value = 1.
          distance.value = 1.3
          Teff_  = 3040.   # Sun-like Teff (K)
          Rs_    = 0.141      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by E. Schwieterman (Meadows et al. 2016)']

       if template.value =='Proxima Cen b 90 bar O2-CO2':
          fn = 'Proxima16_O2_CO2_90bar_prox_hitran2012_50_100000cm_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_ = model[:,0]
          solhr_ = model[:,2]
          Flx_ = model[:,3]
          Ahr_ = Flx_/solhr_
          lamhr_ = lamhr_[::-1]
          Ahr_ = Ahr_[::-1]
          semimajor.value = 0.048
          radius.value = 1.
          distance.value = 1.3
          Teff_  = 3040.   # Sun-like Teff (K)
          Rs_    = 0.141      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by E. Schwieterman (Meadows et al. 2016)']

       if template.value =='Proxima Cen b 90 bar Venus':
          fn = 'Proxima17_smart_spectra_Venus90bar_clouds_500_100000cm-1_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_ = model[:,0]
          solhr_ = model[:,2]
          Flx_ = model[:,3]
          Ahr_ = Flx_/solhr_
          lamhr_ = lamhr_[::-1]
          Ahr_ = Ahr_[::-1]
          semimajor.value = 0.048
          radius.value = 1.
          distance.value = 1.3
          Teff_  = 3040.   # Sun-like Teff (K)
          Rs_    = 0.141      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by G. Arney (Meadows et al. 2016)']

       if template.value =='Proxima Cen b 10 bar Venus':
          fn = 'Proxima17_smart_spectra_Venus10bar_cloudy_500_100000cm-1_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_ = model[:,0]
          solhr_ = model[:,2]
          Flx_ = model[:,3]
          Ahr_ = Flx_/solhr_
          lamhr_ = lamhr_[::-1]
          Ahr_ = Ahr_[::-1]
          semimajor.value = 0.048
          radius.value = 1.
          distance.value = 1.3
          Teff_  = 3040.   # Sun-like Teff (K)
          Rs_    = 0.141      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by G. Arney (Meadows et al. 2016)']

       if template.value =='Proxima Cen b CO2/CO/O2 dry':
          fn = 'Proxima18_gao_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_ = model[:,0]
          solhr_ = model[:,2]
          Flx_ = model[:,3]
          Ahr_ = Flx_/solhr_
          lamhr_ = lamhr_[::-1]
          Ahr_ = Ahr_[::-1]
          semimajor.value = 0.048
          radius.value = 1.
          distance.value = 1.3
          Teff_  = 3040.   # Sun-like Teff (K)
          Rs_    = 0.141      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by E. Schwieterman based on work by P. Gao (Meadows et al. 2016; Gao et al. 2015)']            

       if template.value =='Proxima Cen b Earth':
          # this one needs a weighted average
          fn = 'Proxima19_earth_prox.pt_stratocum_hitran2012_50_100000cm_toa.rad'
          fn1 = 'Proxima19_earth_prox.pt_filtered_hitran2012_50_100000cm_toa.rad'
          fn2 = 'Proxima19_earth_prox.pt_stratocum_hitran2012_50_100000cm_toa.rad'
          fn = os.path.join(relpath, fn)
          fn1 = os.path.join(relpath, fn1)
          fn2 = os.path.join(relpath, fn2)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_ = model[:,0]
          solhr_ = model[:,2]
          Flx_ = model[:,3]
          model1 = np.loadtxt(fn1, skiprows=1)
          lamhr_1 = model1[:,0]
          solhr_1 = model1[:,2]
          Flx_1 = model1[:,3]
          model2 = np.loadtxt(fn2, skiprows=1)
          lamhr_2 = model2[:,0]
          solhr_2 = model2[:,2]
          Flx_2 = model2[:,3]
          Ahr_ = Flx_/solhr_
          Ahr_1 = Flx_1/solhr_1
          Ahr_2 = Flx_2/solhr_2
          Ahr_ = (Ahr_*0.25+Ahr_2*0.25+Ahr_1*0.5)
          lamhr_ = lamhr_[::-1]
          Ahr_ = Ahr_[::-1]
          semimajor.value = 0.048
          radius.value = 1.
          distance.value = 1.3
          Teff_  = 3040.   # Sun-like Teff (K)
          Rs_    = 0.141      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by E. Schwieterman (Meadows et al. 2016)']  

       if template.value =='Proxima Cen b Archean Earth':
          fn = 'Proxima21_HAZE_msun21_0.0Ga_1.00e-02ch4_rmix_5.0E-2__30.66fscale_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_ = model[:,0]
          solhr_ = model[:,2]
          Flx_ = model[:,3]
          Ahr_ = Flx_/solhr_
          lamhr_ = lamhr_[::-1]
          Ahr_ = Ahr_[::-1]
          semimajor.value = 0.048
          radius.value = 1.
          distance.value = 1.3
          Teff_  = 3040.   # Sun-like Teff (K)
          Rs_    = 0.141      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by G. Arney (Meadows et al. 2016)']           

       if template.value =='Proxima Cen b hazy Archean Earth':
          fn = 'Proxima21_HAZE_msun21_0.0Ga_3.00e-02ch4_rmix_5.0E-2__30.66fscale_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_ = model[:,0]
          solhr_ = model[:,2]
          Flx_ = model[:,3]
          Ahr_ = Flx_/solhr_
          lamhr_ = lamhr_[::-1]
          Ahr_ = Ahr_[::-1]
          semimajor.value = 0.048
          radius.value = 1.
          distance.value = 1.3
          Teff_  = 3040.   # Sun-like Teff (K)
          Rs_    = 0.141      # star radius in solar radii
          solhr_ =  cg.noise_routines.Fstar(lamhr_, Teff_, Rs_, semimajor.value, AU=True)
          planet_label = ['Synthetic spectrum generated by G. Arney (Meadows et al. 2016)']           
          
       global lammin
       global lammax
       global planet_label
       lammin=min(lamhr_)
       if lammin <= 0.2:
          lammin = 0.2
       lammax=3.
          
       

    print "ground based = ", ground_based.value
    if ground_based.value == "No":
       ground_based_ = False
    if ground_based.value == "Yes":
       ground_based_ = True
    
    # Run coronagraph 
    lam, dlam, A, q, Cratio, cp, csp, cz, cez, cD, cR, cth, DtSNR = \
        cg.count_rates(Ahr_, lamhr_, solhr_, alpha,  radius.value, Teff_, Rs_, semimajor.value, distance.value, exozodi.value, diam=diameter.value, Res=resolution.value, Res_UV = resolution_UV.value, Res_NIR = resolution_NIR.value, Tsys=temperature.value, IWA=inner.value, OWA=outer.value, lammin=lammin, lammax=lammax, De=De, Re=Re, Dtmax = dtmax.value, THERMAL=True, GROUND=ground_based_, wantsnr=want_snr.value)


    # Calculate background photon count rates
    cb = (cz + cez + csp + cD + cR + cth)
    # Convert hours to seconds
    Dts = exptime.value * 3600.
    # Calculate signal-to-noise assuming background subtraction (the "2")
    SNR  = cp*Dts/np.sqrt((cp + 2*cb)*Dts)
    # Calculate 1-sigma errors
    sig= Cratio/SNR
    # Add gaussian noise to flux ratio
    spec = Cratio + np.random.randn(len(Cratio))*sig
    lastlam = lam
    lastCratio = Cratio
    global lastlam
    global lastCratio

    #UPDATE DATA
    planet.data = dict(lam=lam, cratio=Cratio*1e9, spec=spec*1e9, downerr=(spec-sig)*1e9, uperr=(spec+sig)*1e9, cz=cz*Dts, cez=cez*Dts, csp=csp*Dts, cD=cD*Dts, cR=cR*Dts, cth=cth*Dts, cp=cp*Dts)
    expplanet.data = dict(lam=lam[np.isfinite(DtSNR)], DtSNR=DtSNR[np.isfinite(DtSNR)])
     #make the data the time for a given SNR if user wants this:
    textlabel.data = dict(label=planet_label)

    format_button_group.active = None
    lasttemplate = template.value

    #IF YOU WANT COMPARISON SPECTRUM:
    if comparison.value != lastcomparison:
      if comparison.value == 'Earth':
          fn = 'earth_quadrature_radiance_refl.dat'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=8)
          lamhr_c = model[:,0]
          radhr_c = model[:,1]
          solhr_c = model[:,2]
          Ahr_c   = np.pi*(np.pi*radhr_c/solhr_c)
          semimajor_c = 1.
          radius_c = 1.
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by T. Robinson (Robinson et al. 2011)']

      if comparison.value == 'Venus':
          fn = 'new_venus.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_c = model[:,0]
          Fhr_c = model[:,3]
          solhr_c = model[:,2]
          Ahr_c = (Fhr_c/solhr_c)
          lamhr_c = lamhr_c[::-1]
          Ahr_c = Ahr_c[::-1]
          semimajor_c = 0.72
          radius_c = 0.94
          Teff_c = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by G. Arney']

      if comparison.value =='Archean Earth':
          fn = 'ArcheanEarth_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=8)
          lamhr_c = model[:,0]
          Ahr_c = model[:,1]
          semimajor_c = 1.
          radius_c = 1.
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by G. Arney (Arney et al. 2016)']
          
      if comparison.value =='Hazy Archean Earth':
          fn = 'Hazy_ArcheanEarth_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=8)
          lamhr_c = model[:,0]
          Ahr_c = model[:,1]
          semimajor_c = 1.
          radius_c = 1.
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by G. Arney (Arney et al. 2016)']


      if comparison.value =='1% PAL O2 Proterozoic Earth':
          fn = 'proterozoic_hi_o2_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_c = model[:,0]
          Ahr_c = model[:,1]
          semimajor_c = 1.
          radius_c = 1.
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by G. Arney (Arney et al. 2016)']
          

      if comparison.value =='0.1% PAL O2 Proterozoic Earth':
          fn = 'proterozoic_low_o2_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_c = model[:,0]
          Ahr_c = model[:,1]
          semimajor_c = 1.
          radius_c = 1.
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by G. Arney (Arney et al. 2016)']

          
      if comparison.value =='Early Mars':
          fn = 'EarlyMars_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=8)
          lamhr_c = model[:,0]
          Ahr_c = model[:,1]
          semimajor_c = 1.52
          radius_c = 0.53
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by G. Arney based on Smith et al. 2014']

          
      if comparison.value =='Mars':
          fn = 'Mars_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=8)
          lamhr_c = model[:,0]
          Ahr_c = model[:,1]
          semimajor_c = 1.52
          radius_c = 0.53         
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by T. Robinson']

          
      if comparison.value =='Jupiter':
          fn = 'Jupiter_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_c = model[:,0]
          Ahr_c = model[:,1]
          semimajor_c = 5.46
          radius_c = 10.97
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['0.9-0.3 microns observed by Karkoschka et al. (1998); 0.9-2.4 microns observed by Rayner et al. (2009)']

          
      if comparison.value =='Saturn':
          fn = 'Saturn_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_c = model[:,0]
          Ahr_c = model[:,1]
          semimajor_c = 9.55
          radius_c = 9.14
          Teff_c = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['0.9-0.3 microns observed by Karkoschka et al. (1998); 0.9-2.4 microns observed by Rayner et al. (2009)']

          
      if comparison.value =='Uranus':
          fn = 'Uranus_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_c = model[:,0]
          Ahr_c = model[:,1]
          semimajor_c = 19.21
          radius_c = 3.98
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['0.9-0.3 microns observed by Karkoschka et al. (1998); 0.9-2.4 microns observed by Rayner et al. (2009)']

          
      if comparison.value =='Neptune':
          fn = 'Neptune_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_c = model[:,0]
          Ahr_c = model[:,1]
          semimajor_c = 29.8
          radius_c = 3.86
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['0.9-0.3 microns observed by Karkoschka et al. (1998); 0.9-2.4 microns observed by Rayner et al. (2009)']


      if comparison.value =='Warm Neptune at 2 AU':
          fn = 'Reflection_a2_m1.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_c = model[:,0]
          Ahr_c = model[:,1]
          lamhr_c = lamhr_c / 1000. #convert to microns
          Ahr_c = Ahr_c * 0.67 #convert to geometric albedo
          semimajor_c = 1.0
          radius_c = 3.86
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by R. Hu (Hu and Seager 2014)']

      if comparison.value =='Warm Neptune w/o Clouds at 1 AU':
          fn = 'Reflection_a1_m2.6_LM_NoCloud.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_c = model[:,0]
          Ahr_c = model[:,1]
          lamhr_c = lamhr_c / 1000. #convert to microns
          Ahr_c = Ahr_c* 0.67 #convert to geometric albedo
          semimajor_c = 1.0
          radius_c = 3.86
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by R. Hu (Hu and Seager 2014)']
          
      if comparison.value =='Warm Neptune w/ Clouds at 1 AU':
          fn = 'Reflection_a1_m2.6_LM.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_c = model[:,0]
          Ahr_c = model[:,1]
          lamhr_c = lamhr_c / 1000. #convert to microns
          Ahr_c = Ahr_c * 0.67 #convert to geometric albedo
          semimajor_c = 2.0
          radius_c = 3.86
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by R. Hu']

      if comparison.value =='Warm Jupiter at 0.8 AU':
          fn = '0.8AU_3x.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_c = model[:,1]
          Ahr_c = model[:,3]
          semimajor_c = 0.8
          radius_c = 10.97
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by K. Cahoy (Cahoy et al. 2010)']

      if comparison.value =='Warm Jupiter at 2 AU':
          fn = '2AU_3x.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_c = model[:,1]
          Ahr_c = model[:,3]
          semimajor_c = 2.0
          radius_c = 10.97
          Teff_c  = 5780.   # Sun-like Teff (K)
          Rs_c    = 1.      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by K. Cahoy (Cahoy et al. 2010)']              

      if comparison.value =='False O2 Planet (F2V star)':
          fn = 'fstarcloudy_geo_albedo.txt'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=0)
          lamhr_c = model[:,0]
          Ahr_c = model[:,1]
          semimajor_c = 1.72 #Earth equivalent distance for F star
          radius_c = 1.
          Teff_c  = 7050.   # F2V Teff (K)
          Rs_c    = 1.3     # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by S. Domagal-Goldman (Domagal-Goldman et al. 2014)']          

      if comparison.value =='Proxima Cen b 10 bar 95% O2 dry':
          fn = 'Proxima15_o2lb_10bar_dry.pt_filtered_hitran2012_50_100000cm_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_c = model[:,0]
          solhr_c = model[:,2]
          Flx_c = model[:,3]
          Ahr_c = Flx_c/solhr_c
          lamhr_c = lamhr_c[::-1]
          Ahr_c = Ahr_c[::-1]
          semimajor_c = 0.048
          radius_c = 1.
          distance_c = 1.3
          Teff_c  = 3040.   # Sun-like Teff (K)
          Rs_c    = 0.141      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by E. Schwieterman (Meadows et al. 2016)']

          
      if comparison.value =='Proxima Cen b 10 bar 95% O2 wet':
          fn = 'Proxima15_o2lb_10bar_h2o.pt_filtered_hitran2012_50_100000cm_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_c = model[:,0]
          solhr_c = model[:,2]
          Flx_c = model[:,3]
          Ahr_c = Flx_c/solhr_c
          lamhr_c = lamhr_c[::-1]
          Ahr_c = Ahr_c[::-1]
          semimajor_c = 0.048
          radius_c = 1.
          distance_c=1.3
          Teff_c  = 3040.   # Sun-like Teff (K)
          Rs_c    = 0.141      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by E. Schwieterman (Meadows et al. 2016)']

      if comparison.value =='Proxima Cen b 10 bar O2-CO2':
          fn = 'Proxima16_O2_CO2_10bar_prox_hitran2012_50_100000cm_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_c = model[:,0]
          solhr_c = model[:,2]
          Flx_c = model[:,3]
          Ahr_c = Flx_c/solhr_c
          lamhr_c = lamhr_c[::-1]
          Ahr_c = Ahr_c[::-1]
          semimajor_c = 0.048
          radius_c = 1.
          distance_c = 1.3
          Teff_c  = 3040.   # Sun-like Teff (K)
          Rs_c    = 0.141      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by E. Schwieterman (Meadows et al. 2016)']

      if comparison.value =='Proxima Cen b 90 bar O2-CO2':
          fn = 'Proxima16_O2_CO2_90bar_prox_hitran2012_50_100000cm_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_c = model[:,0]
          solhr_c = model[:,2]
          Flx_c = model[:,3]
          Ahr_c = Flx_c/solhr_c
          lamhr_c = lamhr_c[::-1]
          Ahr_c = Ahr_c[::-1]
          semimajor_c = 0.048
          radius_c = 1.
          distance_c = 1.3
          Teff_c  = 3040.   # Sun-like Teff (K)
          Rs_c    = 0.141      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by E. Schwieterman (Meadows et al. 2016)']

      if comparison.value =='Proxima Cen b 90 bar Venus':
          fn = 'Proxima17_smart_spectra_Venus90bar_clouds_500_100000cm-1_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_c = model[:,0]
          solhr_c = model[:,2]
          Flx_c = model[:,3]
          Ahr_c = Flx_c/solhr_c
          lamhr_c = lamhr_c[::-1]
          Ahr_c = Ahr_c[::-1]
          semimajor_c = 0.048
          radius_c = 1.
          distance_c = 1.3
          Teff_c  = 3040.   # Sun-like Teff (K)
          Rs_c    = 0.141      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by G. Arney (Meadows et al. 2016)']

      if comparison.value =='Proxima Cen b 10 bar Venus':
          fn = 'Proxima17_smart_spectra_Venus10bar_cloudy_500_100000cm-1_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_c = model[:,0]
          solhr_c = model[:,2]
          Flx_c = model[:,3]
          Ahr_c = Flx_c/solhr_c
          lamhr_c = lamhr_c[::-1]
          Ahr_c = Ahr_c[::-1]
          semimajor_c = 0.048
          radius_c = 1.
          distance_c = 1.3
          Teff_c  = 3040.   # Sun-like Teff (K)
          Rs_c    = 0.141      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by G. Arney (Meadows et al. 2016)']

      if comparison.value =='Proxima Cen b CO2/CO/O2 dry':
          fn = 'Proxima18_gao_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_c = model[:,0]
          solhr_c = model[:,2]
          Flx_c = model[:,3]
          Ahr_c = Flx_c/solhr_c
          lamhr_c = lamhr_c[::-1]
          Ahr_c = Ahr_c[::-1]
          semimajor_c = 0.048
          radius_c = 1.
          distance_c = 1.3
          Teff_c  = 3040.   # Sun-like Teff (K)
          Rs_c    = 0.141      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by E. Schwieterman (Meadows et al. 2016)']            

      if comparison.value =='Proxima Cen b Earth':
          # this one needs a weighted average
          fn = 'Proxima19_earth_prox.pt_stratocum_hitran2012_50_100000cm_toa.rad'
          fn1 = 'Proxima19_earth_prox.pt_filtered_hitran2012_50_100000cm_toa.rad'
          fn2 = 'Proxima19_earth_prox.pt_stratocum_hitran2012_50_100000cm_toa.rad'
          fn = os.path.join(relpath, fn)
          fn1 = os.path.join(relpath, fn1)
          fn2 = os.path.join(relpath, fn2)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_c = model[:,0]
          solhr_c = model[:,2]
          Flx_c = model[:,3]
          model1 = np.loadtxt(fn1, skiprows=1)
          lamhr_1c = model1[:,0]
          solhr_1c = model1[:,2]
          Flx_1c = model1[:,3]
          model2 = np.loadtxt(fn2, skiprows=1)
          lamhr_2c = model2[:,0]
          solhr_2c = model2[:,2]
          Flx_2c = model2[:,3]
          Ahr_c = Flx_c/solhr_c
          Ahr_1c = Flx_1c/solhr_1c
          Ahr_2c = Flx_2c/solhr_2c
          Ahr_c = (Ahr_c*0.25+Ahr_2c*0.25+Ahr_1c*0.5)
          lamhr_c = lamhr_c[::-1]
          Ahr_c = Ahr_c[::-1]
          semimajor_c = 0.048
          radius_c = 1.
          distance_c = 1.3
          Teff_c  = 3040.   # Sun-like Teff (K)
          Rs_c    = 0.141      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by E. Schwieterman (Meadows et al. 2016)']  

      if comparison.value =='Proxima Cen b Archean Earth':
          fn = 'Proxima21_HAZE_msun21_0.0Ga_1.00e-02ch4_rmix_5.0E-2__30.66fscale_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_c = model[:,0]
          solhr_c = model[:,2]
          Flx_c = model[:,3]
          Ahr_c = Flx_c/solhr_c
          lamhr_c = lamhr_c[::-1]
          Ahr_c = Ahr_c[::-1]
          semimajor_c = 0.048
          radius_c = 1.
          distance_c = 1.3
          Teff_c  = 3040.   # Sun-like Teff (K)
          Rs_c    = 0.141      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by G. Arney (Meadows et al. 2016)']           

      if comparison.value =='Proxima Cen b hazy Archean Earth':
          fn = 'Proxima21_HAZE_msun21_0.0Ga_3.00e-02ch4_rmix_5.0E-2__30.66fscale_toa.rad'
          fn = os.path.join(relpath, fn)
          model = np.loadtxt(fn, skiprows=1)
          lamhr_c = model[:,0]
          solhr_c = model[:,2]
          Flx_c = model[:,3]
          Ahr_c = Flx_c/solhr_c
          lamhr_c = lamhr_c[::-1]
          Ahr_c = Ahr_c[::-1]
          semimajor_c = 0.048
          radius_c = 1.
          distance_c = 1.3
          Teff_c  = 3040.   # Sun-like Teff (K)
          Rs_c    = 0.141      # star radius in solar radii
          solhr_c =  cg.noise_routines.Fstar(lamhr_c, Teff_c, Rs_c, semimajor_c, AU=True)
          planet_label_c = ['Synthetic spectrum generated by G. Arney (Meadows et al. 2016)']           

          
      global lammin_c
      global lammax_c
      lammin_c=min(lamhr_c)
      if lammin_c <= 0.2:
         lammin_c = 0.2
      lammax_c=3.

              

    if comparison.value != 'none':
      print 'comparison.value =', comparison.value
      print  'running comparison spectrum'
      try:
         distance_c
      except NameError:
         lamC, dlamC, AC, qC, CratioC, cpC, cspC, czC, cezC, cDC, cRC, cthC, DtSNRC = \
       cg.count_rates(Ahr_c, lamhr_c, solhr_c, alpha,  radius_c, Teff_c, Rs_c, semimajor_c, distance.value, exozodi.value, diam=diameter.value, Res=resolution.value, Res_UV = resolution_UV.value, Res_NIR = resolution_NIR.value,Tsys=temperature.value, IWA=inner.value, OWA=outer.value, lammin=lammin, lammax=lammax, De=De, Re=Re, Dtmax = dtmax.value, THERMAL=True, GROUND=ground_based_, wantsnr=want_snr.value)
      else:    
         lamC, dlamC, AC, qC, CratioC, cpC, cspC, czC, cezC, cDC, cRC, cthC, DtSNRC = \
       cg.count_rates(Ahr_c, lamhr_c, solhr_c, alpha, radius_c, Teff_c, Rs_c, semimajor_c, distance_c, exozodi.value, diam=diameter.value, Res=resolution.value, Res_UV = resolution_UV.value, Res_NIR = resolution_NIR.value,Tsys=temperature.value, IWA=inner.value, OWA=outer.value, lammin=lammin, lammax=lammax, De=De, Re=Re, Dtmax = dtmax.value, THERMAL=True, GROUND=ground_based_, wantsnr=want_snr.value)


      

    if comparison.value == 'none':
       lamC = lamhr_ * 0.
       CratioC = Ahr_ *0.
       DtSNRC = DtSNR * 0.


    lastcomparison = comparison.value

    #UPDATE DATA
    compare.data = dict(lam=lamC, cratio=CratioC*1e9)
    expcompare.data = dict(lam=lamC[np.isfinite(DtSNRC)], DtSNR=DtSNRC[np.isfinite(DtSNRC)])
        
    #######PLOT UPDATES#######    
    global snr_ymax_
    global snr_ymin_

    ii = np.where(lam < 2.5) #only want where reflected light, not thermal
    iii = np.where(lamC < 2.5)  #only want where reflected light, not thermal
   # pdb.set_trace()
    Cratio_ok = Cratio[ii]
    CratioC_ok = CratioC[iii]
    Cratio_ok = Cratio_ok[~np.isnan(Cratio_ok)]
    CratioC_ok = CratioC_ok[~np.isnan(CratioC_ok)]
    print 'snr_ymax_',  np.max([np.max(Cratio_ok)*1e9, np.max(CratioC_ok)*1e9])
    print 'snr_ymin_',  np.min([np.min(Cratio_ok)*1e9, np.min(CratioC_ok)*1e9])
    snr_ymax_ = np.max([np.max(Cratio_ok)*1e9*1.5, np.max(CratioC_ok)*1e9*1.5])
    snr_ymin_ = np.min([np.min(CratioC_ok)*1e9, np.min(CratioC_ok)*1e9])
  #  snr_plot.y_range.start = -0.2

    exp_plot.yaxis.axis_label='Integration time for SNR = '+str(want_snr.value)+' [hours]' 

    
    if comparison.value != 'none':
       snr_plot.title.text = 'Planet Spectrum: '+template.value +' and comparison spectrum '+comparison.value
       exp_plot.title.text = 'Planet Spectrum: '+template.value +' and comparison spectrum '+comparison.value
       
    if comparison.value == 'none':
      snr_plot.title.text = 'Planet Spectrum: '+template.value
      exp_plot.title.text =  'Planet Spectrum: '+template.value

    if template.value == 'Early Mars' or template.value == 'Mars':
       if comparison.value == 'none' or comparison.value == 'Early Mars' or comparison.value == 'Mars':
          snr_plot.y_range.end = snr_ymax_ + 2.*snr_ymax_
    else:
       snr_plot.y_range.end = snr_ymax_ + 0.2*snr_ymax_


    

       
######################################
# SET UP ALL THE WIDGETS AND CALLBACKS 
######################################

source = ColumnDataSource(data=dict(value=[]))
source.on_change('data', update_data)
exptime  = Slider(title="Integration Time (hours)", value=24., start=1., end=1000.0, step=1.0, callback_policy='mouseup')
exptime.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
distance = Slider(title="Distance (parsec)", value=10., start=1.28, end=50.0, step=0.2, callback_policy='mouseup') 
distance.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
radius   = Slider(title="Planet Radius (R_Earth)", value=1.0, start=0.5, end=20., step=0.1, callback_policy='mouseup') 
radius.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
semimajor= Slider(title="Semi-major axis of orbit (AU)", value=1.0, start=0.01, end=20., step=0.01, callback_policy='mouseup') 
semimajor.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
exozodi  = Slider(title="Number of Exozodi", value = 3.0, start=1.0, end=10., step=1., callback_policy='mouseup') 
exozodi.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
diameter  = Slider(title="Mirror Diameter (meters)", value = 10.0, start=0.5, end=50., step=0.5, callback_policy='mouseup') 
diameter.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
resolution  = Slider(title="Telescope Visible Resolution (R)", value = 150.0, start=10.0, end=300., step=5., callback_policy='mouseup') 
resolution.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
resolution_UV  = Slider(title="Telescope UV Resolution (R)", value = 20.0, start=10.0, end=300., step=5., callback_policy='mouseup') 
resolution_UV.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
resolution_NIR  = Slider(title="Telescope NIR Resolution (R)", value = 100.0, start=10.0, end=300., step=5., callback_policy='mouseup') 
resolution_NIR.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
temperature  = Slider(title="Telescope Temperature (K)", value = 270.0, start=90.0, end=400., step=10., callback_policy='mouseup') 
temperature.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
inner  = Slider(title="Inner Working Angle factor x lambda/D", value = 2.0, start=1.22, end=4., step=0.2, callback_policy='mouseup') 
inner.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
outer  = Slider(title="Outer Working Angle factor x lambda/D", value = 30.0, start=20, end=100., step=1, callback_policy='mouseup') 
outer.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
darkcurrent  = Slider(title="Dark current (counts/s)", value = 1e-4, start=1e-5, end=1e-3, step=1e-5, callback_policy='mouseup') 
darkcurrent.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
readnoise  = Slider(title="Read noise (counts/pixel)", value = 0.1, start=0.01, end=1, step=0.05, callback_policy='mouseup') 
readnoise.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
dtmax  = Slider(title="Maximum single exposure time (hours)", value = 1, start=0.1, end=10., step=0.5, callback_policy='mouseup') 
dtmax.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
want_snr  = Slider(title="Desired signal-to-noise ratio? (only used for exposure time plot)", value = 10, start=0.5, end=100., step=0.5, callback_policy='mouseup') 
want_snr.callback = CustomJS(args=dict(source=source), code="""
    source.data = { value: [cb_obj.value] }
""")
#ground based choice
ground_based = Select(title="Simulate ground-based observation?", value="No", options=["No",  "Yes"])

#select menu for planet
template = Select(title="Planet Spectrum", value="Earth", options=["Earth",  "Archean Earth", "Hazy Archean Earth", "1% PAL O2 Proterozoic Earth", "0.1% PAL O2 Proterozoic Earth","Venus", "Early Mars", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",'----','Warm Neptune at 2 AU', 'Warm Neptune w/o Clouds at 1 AU', 'Warm Neptune w/ Clouds at 1 AU','Warm Jupiter at 0.8 AU', 'Warm Jupiter at 2 AU',"False O2 Planet (F2V star)", '-----', 'Proxima Cen b 10 bar 95% O2 dry', 'Proxima Cen b 10 bar 95% O2 wet', 'Proxima Cen b 10 bar O2-CO2', 'Proxima Cen b 90 bar O2-CO2', 'Proxima Cen b 90 bar Venus', 'Proxima Cen b 10 bar Venus', 'Proxima Cen b CO2/CO/O2 dry', 'Proxima Cen b Earth', 'Proxima Cen b Archean Earth', 'Proxima Cen b hazy Archean Earth' ])
#select menu for comparison spectrum
comparison = Select(title="Show comparison spectrum?", value ="none", options=["none", "Earth",  "Archean Earth", "Hazy Archean Earth", "1% PAL O2 Proterozoic Earth", "0.1% PAL O2 Proterozoic Earth","Venus", "Early Mars", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",'----','Warm Neptune at 2 AU', 'Warm Neptune w/o Clouds at 1 AU', 'Warm Neptune w/ Clouds at 1 AU','Warm Jupiter at 0.8 AU', 'Warm Jupiter at 2 AU', "False O2 Planet (F2V star)", '-----', 'Proxima Cen b 10 bar 95% O2 dry', 'Proxima Cen b 10 bar 95% O2 wet', 'Proxima Cen b 10 bar O2-CO2', 'Proxima Cen b 90 bar O2-CO2', 'Proxima Cen b 90 bar Venus', 'Proxima Cen b 10 bar Venus', 'Proxima Cen b CO2/CO/O2 dry', 'Proxima Cen b Earth', 'Proxima Cen b Archean Earth', 'Proxima Cen b hazy Archean Earth'])


oo = column(children=[exptime, diameter, resolution_UV, resolution, resolution_NIR, temperature, ground_based]) 
pp = column(children=[template, comparison, distance, radius, semimajor, exozodi]) 
qq = column(children=[instruction0, text_input, instruction1, format_button_group, instruction2, link_box])
ii = column(children=[inner, outer,  dtmax])
ee = column(children=[want_snr])

observation_tab = Panel(child=oo, title='Observation')
planet_tab = Panel(child=pp, title='Planet')
instrument_tab = Panel(child=ii, title='Instrumentation')
download_tab = Panel(child=qq, title='Download')
time_tab = Panel(child=ee, title='Exposure Time Calculator')

for w in [text_input]: 
    w.on_change('value', change_filename)
format_button_group.on_click(i_clicked_a_button)

for ww in [template]: 
    ww.on_change('value', update_data)

for www in [comparison]: 
    www.on_change('value', update_data)

for gg in [ground_based]: 
    gg.on_change('value', update_data)


inputs = Tabs(tabs=[ planet_tab, observation_tab, instrument_tab, time_tab, download_tab ])

curdoc().add_root(row(inputs, ptabs)) 
