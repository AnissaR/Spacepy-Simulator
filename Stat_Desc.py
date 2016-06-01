## This code aims at reproducing the graphs of Brautigam and Albert (check the variables definitions)

# Import packages
from spacepy import pycdf
import spacepy
import matplotlib
import matplotlib.pyplot as plt
import spacepy.datamodel as dm
import spacepy.plot as plotty
import numpy as np
import os
import sys
import math
import scipy.constants as const  # physical constants
from scipy.interpolate import griddata
import datetime
import spacepy.omni as omni
import time
from spacepy import coordinates as coord
import spacepy.time as tim
import spacepy.LANLstar as LS
import pandas as pd
from matplotlib.colors  import LogNorm 
from matplotlib.ticker  import (LogLocator, LogFormatter,
                                        LogFormatterMathtext) 
import numpy.ma as ma

###################
# Import physics constants (for unit conversion)
###################

m_e=const.physical_constants['electron mass energy equivalent in MeV'][0]    # electron mass energy equivalent in MeV.c^-2
c=const.c            # speed of light (just in case, m.s**-1)
Re=6378.137             # Earth Radius


###################
# Import databases
###################

### CRRES database : cdf (flux etc data). At unit level, possible to have 500ms precision (but binary files)
os.chdir("/home/anissa/Bureau/Stage/Code CWI/Data/")
cdf = pycdf.CDF('crres_h0_mea_19901006_v01.cdf') # CRRES Database
Epoch=cdf['Epoch'][:]      # vector of time of the CRRES Database
print Epoch
ticks = tim.Ticktock(Epoch, 'UTC')
#print(cdf)

### OMNI database : data_omni (Kp, Dst etc data) for the same dates
# spacepy.toolbox.update(all=True)
data_omni=omni.get_omni(Epoch)
#data_omni.tree(levels=1)

### CRRES Orbit Survey database (start time of CRRES orbit number data)
orbit_survey=pd.read_csv('crres_orbit_survey_wFormat.csv') # Use orbit_survey (in csv file), dates of this file, and Epoch in cdf - CAREFUL : delete the line of the 183. orbit, date problem !

### import factor for determining outer boundary condition
factor_boundaries=pd.read_csv('LANL_satellite.csv') # 4 energy channels are considered : Col1 for mu=100, Col2 for mu=200 - 316, Col3 ofr mu=501, Col5 for mu=794 - 1000
boundary_factor=pd.Series(factor_boundaries['Col1'], index=factor_boundaries['DOY'])

###################
# Build Database with Pandas
###################

### With CRRES database, extract position and time of the satellite (root of the database)
# define coordonates of the satellite
#pycdf.gAttrList(cdf)
#print cdf['Altitude'].attrs
Altitude=cdf['Altitude'][:]         # Altitude of the satellite (km) - coherent with perigee and apogee orbit type in description
radial_distance=Altitude/Re + 1             # radial distance of the satellite (R_e)
#print cdf['Latitude'].attrs
Latitude=cdf['Latitude'][:]         # Latitude of the satellite (degree) - is it in the GEO system ? - coherent with inclination orbit type in description
#print cdf['Longitude'].attrs
Longitude=cdf['Longitude'][:]         # Longitude of the satellite (degree) - is it in the GEO system ?

### Add survey data to find orbit number
temp1 = np.array(pd.DataFrame(orbit_survey, columns=['ORBIT'])).flatten().astype(float)
temp2=[tim.doy2date(1900+np.array(pd.DataFrame(orbit_survey, columns=['YR'])).flatten()[i],np.array(pd.DataFrame(orbit_survey, columns=['DOY'])).flatten()[i],dtobj=True)
 + tim.sec2hms(np.array(pd.DataFrame(orbit_survey, columns=['START'])).flatten()[i],rounding=False,days=False,dtobj=True) for i in range(len(temp1))]
orbit_dates = pd.Series(temp1, index=temp2)
orbit_dates = orbit_dates[~orbit_dates.index.duplicated(keep='first')]
orbit_epoch = pd.Series(np.nan, index=Epoch)
orbit_epoch = orbit_epoch[~orbit_epoch.index.duplicated(keep='first')]
# interpolate the orbits according to these two vectors
all_orbits=pd.concat([orbit_dates, orbit_epoch]).sort_index()
all_orbits = all_orbits[~all_orbits.index.duplicated(keep='first')]
all_orbits=all_orbits.interpolate(method='time')[Epoch]
orbit_number=np.array(pd.DataFrame(all_orbits, columns=['0'])).flatten()
#plt.plot(Epoch,orbit_number)
del temp1
del temp2
del orbit_dates
del orbit_epoch
del all_orbits

### Create panda dataframe with all data for Epoch time
index=range(len(Altitude))
temp=np.array([[Epoch[i],orbit_number[i],radial_distance[i],Latitude[i],Longitude[i]] for i in index])    # Satellite coordonates in Geographic Coordinate System (GEO) ? As radial dist (re), latitude (degree), lomgitude (degree)
database = pd.DataFrame(temp,index=index,columns=['Epoch (datetime)','Orbit number','Radial Distance (Re)','Latitude (degree)','Longitude (degree)'])
#database.loc[718,['Orbit number']]
#test=database.loc[:,['Orbit number']]
del temp


### Call fixed parameters of angle and Energy

energy_of_flux=cdf['energy_of_flux'][:]     # considered energy channel of the MEA
energy=cdf['energy'][:]                     # same but numeric (keV)
energy_MeV=energy*10**(-3)                  # same in MeV

angle_of_flux=cdf['angle_of_flux'][:]     # considered angle of the MEA
angle=cdf['angle'][:]                     # same but numeric
angle_rad=angle*math.pi/180               # same in radian


### Calculate L* using lanlstar (external ;agn field : T89. No hypothesis on the internal (measured by Bmirr ??))
# for inputdict with T89: ['Year', 'DOY', 'Hr', 'Kp', 'Pdyn', 'ByIMF', 'BzIMF', 'Lm', 'Bmirr', 'PA', 'rGSM', 'latGSM', 'lonGSM']
inputdict = {}
# Time data
inputdict['Year']   = np.float32([date.year for date in Epoch])
inputdict['DOY']    = np.float32(ticks.DOY)
inputdict['Hr']     = ((3600*np.float32([date.hour for date in Epoch])+60*np.float32([date.minute for date in Epoch])+np.float32([date.second for date in Epoch]))/3600)
# OMNI data
inputdict['Kp']     = np.array(data_omni['Kp'])            # Kp index
inputdict['Dst']    = np.array(data_omni['Dst'])            # Dst index (nT) (Normally, not used in T89, but doesn't run without it ???)
inputdict['Pdyn']   = np.array(data_omni['Pdyn'])            # solar wind dynamic pressure (nPa)
inputdict['ByIMF']  = np.array(data_omni['ByIMF'])            # GSM y component of IMF magnetic field (nT)
inputdict['BzIMF']  = np.array(data_omni['BzIMF'])            # GSM z component of IMF magnetic field (nT)
# CRRES data
inputdict['Lm']     = np.array(cdf['L'])             # McIllwain L Value according CRRES
inputdict['Bmirr']  = np.array(cdf['B'])             # magnetic field strength at the mirror point - in crres database, B value at mirror poimt (ie alpha=90 degree) - is it the good variable ?
# for satellite coordonates need to pass on GSM coordinate system
coordonnees = [[radial_distance[i],Latitude[i],Longitude[i]] for i in index]
coordonnees = coord.Coords(coordonnees, 'GEO', 'sph') # define coordonates as class ([rad, lat, lon])
coordonnees.ticks=ticks
coordonnees.units=['Re','deg','deg']
coord_GSM=coordonnees.convert('GSM','sph')  # coordonate of the satellite on GSM system, as [rad,lat,lon] (good units ?)
inputdict['rGSM']   = coord_GSM.data[:,0]             # radial coordinate in GSM [Re]
inputdict['latGSM'] = coord_GSM.data[:,1]             # latitude coordiante in GSM [deg]
inputdict['lonGSM'] = coord_GSM.data[:,2]             # longitude coodrinate in GSM [deg]
# For each angle aned time step, calculate the Lstar
nb_time=len(index)
nb_angle=len(angle)
invariant_Lstar=np.zeros([nb_time,nb_angle])      # Matrix containing values of L* for time x angles of CRRES database
for i in range(nb_angle):
    inputdict['PA']=[angle[i]]*nb_time            # pitch angle [deg]
    invariant_Lstar[:,i]=LS.LANLstar(inputdict, 'T89')['T89']            # for each angle, for all times, calculate faster than irbempy.get_Lstar , gives similar but not equal results
### NOW WE HAVE L* VALUES FOR EACH TIME AND EACH ANGLE

''' # FOR CHECKING inputdict VALUES AND UNITS (not needed for compilation)
# 
plt.plot(inputdict['DOY']+inputdict['Hr']/24,inputdict['Kp']) # look like the article (fig2), no units
plt.plot(inputdict['DOY']+inputdict['Hr']/24,inputdict['Dst']) # look like the article (fig2), in nT
plt.plot(inputdict['DOY']+inputdict['Hr']/24,inputdict['Pdyn']) # not in the article with these units (cf fig1), in nPa
plt.plot(inputdict['DOY']+inputdict['Hr']/24,inputdict['ByIMF']) # not in the article (fig1), in nT
plt.plot(inputdict['DOY']+inputdict['Hr']/24,inputdict['BzIMF']) # look like the article (fig1), in nT
plt.plot(inputdict['DOY']+inputdict['Hr']/24,inputdict['Lm']) # not in the article, no units
#print cdf['L'].attrs
plt.plot(inputdict['DOY']+inputdict['Hr']/24,inputdict['Bmirr']) # not in the article, in nT (good unit for LANLstar ?)
#print cdf['B'].attrs
plt.plot(inputdict['DOY']+inputdict['Hr']/24,inputdict['rGSM']) # not in the article, in Re
plt.plot(inputdict['DOY']+inputdict['Hr']/24,inputdict['latGSM']) # not in the article, in degree
plt.plot(inputdict['DOY']+inputdict['Hr']/24,inputdict['lonGSM']) # not in the article, in degree
'''
### IS IT OK ?


### Import data of electrons fluxes
Flux=cdf['FLUX'][:,:,:]                     # MEA Flux (in 1/[cm**2.s.sr.keV]). 1st variable, time, 2nd variable, angle, 3rd variable, energy
#print cdf['FLUX'].attrs
Flux=Flux*10**3                             # Convert MEA Flux in 1/[cm**2.s.sr.MeV]
# Plot if you want fluxes for a given level of energy, depending on time and pitch angle
heatmap = plt.pcolor(orbit_number, angle, np.transpose(Flux[:,:,10]), cmap='RdBu', vmin=10.0**2, vmax=10**7,norm=LogNorm())
plt.colorbar(heatmap)
plt.axis([orbit_number.min(), orbit_number.max(), angle.min(), angle.max()])


'''
REPRODUCE Plate 1 : Suppose fixed energy and second invariant (as in Albert), plot MEA flux for given energy channel depending on orbit
'''

#print energy_MeV

E_index=3
E=energy_MeV[E_index]            # Energy fixed in 0.34 MeV
# relativistic momentum, derived from (pc)²=(E+E0)²-E0², can be used for representing PSD

flux=np.zeros([nb_time,nb_angle])  # phase space density for fixed M
for i in range(0, nb_time):
     for j in range(0, nb_angle):
         flux[i,j]=Flux[i,j,E_index]
         if flux[i,j]<0:
             flux[i,j]=np.nan


bin_Lstar=np.arange(3,6.55,0.1)
nbin=len(bin_Lstar)
orbits=np.arange(180,200.1,0.5)
nb_orbits=len(orbits)
sX_interp, sY_interp = np.meshgrid(orbits, bin_Lstar)  # coordonates in which we interpolate

# data in which we know the values
sX=np.transpose(np.array([orbit_number]*nb_angle)).flatten()
sY=invariant_Lstar.flatten()
sValues=flux.flatten()

# interpolate in the right grid
heatmap=griddata((sX,sY),sValues,(sX_interp, sY_interp))

heatmap = ma.masked_where(np.isnan(heatmap),heatmap)
plot=plt.pcolor(orbits, bin_Lstar, heatmap, cmap='rainbow' , vmin=10.0**3, vmax=10**7,norm=LogNorm())
plt.colorbar(plot)
plt.axis([orbits.min(), orbits.max(), bin_Lstar.min(), bin_Lstar.max()])








'''
REPRODUCE Plate 2 : Suppose fixed first and second invariant (as in Albert), plot PSD value for given first invariant depending on orbit and L*
'''


# Suppose fixed first and second invariant (as in Albert), deduce p
invariant_M=100             # first invariant, fixed at M=100 MeV/G
invariant_J=1.78*10**(-16)  # second invariant, fixed at J=1.78*10**-16 g(cm/s)Re
B_G=np.array(cdf['B'])*10**(-5)                                      # Bmirr in Gauss


### A RECODER
p_momentum=np.zeros([nb_time,nb_angle])  # relativistic momentum
Energy_for_p=np.zeros([nb_time,nb_angle])  # deduce level of energy
Flux_nan=Flux
Flux_nan[np.where(Flux<0)]=np.nan
flux_interpolated=np.zeros([nb_time,nb_angle])  # corresponding interpolated fluxes
psd=np.zeros([nb_time,nb_angle])  # phase space density for fixed M
for i in range(0, nb_time):
     for j in range(0, nb_angle):
         p_momentum[i,j]=math.sqrt(2*m_e*B_G[i]/math.sin(angle_rad[j])**2*invariant_M)
         Energy_for_p[i,j]=math.sqrt(p_momentum[i,j]**2+m_e**2)-m_e             # Comes from (pc)²+E0²=(E+E0)²
         temp=Flux_nan[i,j,:]
         flux_interpolated[i,j]=np.interp(Energy_for_p[i,j],energy_MeV,temp)
         psd[i,j]=flux_interpolated[i,j]/p_momentum[i,j]**2

bin_Lstar=np.arange(3,6.55,0.1)
nbin=len(bin_Lstar)
orbits=np.arange(180,200.1,0.5)
nb_orbits=len(orbits)
sX_interp, sY_interp = np.meshgrid(orbits, bin_Lstar)  # coordonates in which we interpolate

# data in which we know the values
sX=np.transpose(np.array([orbit_number]*nb_angle)).flatten()
sY=invariant_Lstar.flatten()
sValues=psd.flatten()

# interpolate in the right grid
heatmap=griddata((sX,sY),sValues,(sX_interp, sY_interp))

heatmap = ma.masked_where(np.isnan(heatmap),heatmap)
plot=plt.pcolor(orbits, bin_Lstar, heatmap, cmap='rainbow' , vmin=10.0**3, vmax=10**7,norm=LogNorm())
plt.colorbar(plot)
plt.axis([orbits.min(), orbits.max(), bin_Lstar.min(), bin_Lstar.max()])




#### OR
'''


#np.amin(invariant_Lstar)
#np.amax(invariant_Lstar)
bin_Lstar=np.arange(3,6.7,0.1)
nbin=len(bin_Lstar)

orbits=np.array(list(set(orbit_number)))
nb_orbits=len(set(orbit_number))


data_for_spectrogram=np.zeros([nb_orbits,nbin-1])  # phase space density for fixed M
unnan_psd=psd
unnan_psd[np.isnan(unnan_psd)]=0
for i in range(nbin-1):
    for k in range(nb_orbits):
        in_orbit=np.transpose(np.array([(orbit_number==orbits[k])*1,]*nb_angle))
        not_nan=1-np.isnan(psd)
        in_bin=(invariant_Lstar >= bin_Lstar[i])*(invariant_Lstar <bin_Lstar[i+1])*in_orbit*not_nan
        num_by_orbit=sum(sum(in_bin))
        average_val=sum(sum(in_bin*unnan_psd))/num_by_orbit
        data_for_spectrogram[k,i]=average_val


heatmap = plt.pcolor(data_for_spectrogram)
plt.colorbar(heatmap)


x=DOY
y=bin_Lstar[0:36]
heatmap=data_for_spectrogram
extent = [x[0], x[-1], y[0], y[-1]]

plt.clf()
plt.imshow(heatmap, extent=extent)
plt.show()



# Histogram of Lstar (all time)
hist, bin_edges = np.histogram(invariant_Lstar, bins=bin_Lstar)
plt.bar(bin_edges[:-1], hist, width=0.1) 

# BoxPlot of Lstar (depending on time)
plt.boxplot(np.transpose(invariant_Lstar))

# plot altitude depending on time (find orbits ?)
temp1=cdf['MLT'][:]
temp2=cdf['Altitude'][:]
temp3=cdf['Latitude'][211:799]
temp4=cdf['Longitude'][211:799]
test_time=cdf['Epoch'][211:799]
plt.plot(temp1,temp2)
plt.plot(temp1,temp3)
plt.plot(temp1,temp4)
























'''



data_for_spectrogram=np.zeros([nb_orbits,nbin-1])  # phase space density for fixed M
unnan_flux=flux
unnan_flux[np.isnan(unnan_flux)]=0
for i in range(nbin-1):
    for k in range(nb_orbits):
        in_orbit=np.transpose(np.array([(orbit_number==orbits[k])*1,]*nb_angle))
        not_nan=1-np.isnan(flux)
        in_bin=(invariant_Lstar >= bin_Lstar[i])*(invariant_Lstar <bin_Lstar[i+1])*in_orbit*not_nan
        num_by_orbit=sum(sum(in_bin))
        average_val=sum(sum(in_bin*unnan_flux))/num_by_orbit
        data_for_spectrogram[k,i]=average_val


heatmap = plt.pcolor(data_for_spectrogram)
plt.colorbar(heatmap)


# Plot phase space density, masking out values of 0.
plotty=np.transpose(np.where(data_for_spectrogram > 0.0, data_for_spectrogram, 10.0**-39))
map = plt.pcolor(orbits, bin_Lstar[0:36],plotty,vmin=10.0**2, vmax=10.0**10,
                     norm=LogNorm())
cbar = plt.colorbar(map, pad=0.01, shrink=.85, ticks=LogLocator(),
                          format=LogFormatterMathtext())
cbar.set_label('Phase Space Density') 




ax1.set_ylabel('L*')




                     










         
         
'''













######################################################### 
#########################################################
#########################################################
#########################################################
#########################################################
#########################################################









#del inputdict

###
# Fix 1st invariant M to preassigned values and deduce momentum p from alpha and B
# Deduce J and Energy
###

first_invariant_M=np.logspace(math.log(20,10), math.log(1258,10), num=19)
                           # 19 logarithmically spaced values from 20 to 1258 (MeV/G)
B=cdf['B'][:]                               # Local field magnitude (nT)
B_G=B*10**(-5)                                      # same in Gauss
#### a supprimer
#B_G = B_G[0:9]
#### fin de suppression
nb_obs=len(Lstar_invariant[:,0])                    # observations
nb_angles=len(Lstar_invariant[0,:])                    # angles
nb_M=len(first_invariant_M)

p_momentum=np.zeros(shape=(nb_obs,nb_angles,nb_M))  # relativistic
momentum (MeV.c**-1)
second_invariant_J=np.zeros(shape=(nb_obs,nb_angles,nb_M))  # second invariant J (? R_e ?) - Should I fix it ???
Energy_for_p=np.zeros(shape=(nb_obs,nb_angles,nb_M))    # E(p) (MeV), derived from the relativistic momentum-energy relation
First_invariant_rep=np.zeros(shape=(nb_obs,nb_angles,nb_M))    # reapeated first invariant

for i in range(0, nb_obs):
     for j in range(0, nb_angles):
         for k in range(0, nb_M):
             p_momentum[i,j,k]=math.sqrt(2*m_e*B_G[i]/math.sin(angle_rad[j])**2*first_invariant_M[k])
             second_invariant_J[i,j,k]=2*p_momentum[i,j,k]*I_invariant[i,j]
             Energy_for_p[i,j,k]=math.sqrt(p_momentum[i,j,k]**2+m_e**2)-m_e
             First_invariant_rep[i,j,k]=first_invariant_M[k]

# On recupere les flux d'electrons pour le niveau d'energie correspondant (interpolation lineaire)
FLUX=cdf['FLUX'][:,:,:]                     # MEA Flux (in 1/[cm**2.s.sr.keV]). 1st variable, time, 2nd variable, angle, 3rd variable, energy
print cdf['FLUX'].attrs
FLUX=FLUX*10**3                             # Convert MEA Flux in 1/[cm**2.s.sr.MeV]
#### a supprimer
#FLUX=FLUX[0:9,16:19,:]
#### fin de suppression
# for each point, interpolate the flux from the energy channels and deduce the PSD
flux_interpolated=np.zeros(shape=(nb_obs,nb_angles,nb_M)) # interpolated MEA Flux in 1/[cm**2.s.sr.MeV]
PSD=np.zeros(shape=(nb_obs,nb_angles,nb_M))             # phase space density
for i in range(0, nb_obs):
     for j in range(0, nb_angles):
         for k in range(0, nb_M):
             temp=FLUX[i,j,:]
             flux_interpolated[i,j,k]=np.interp(Energy_for_p[i,j,k],energy_MeV,temp)
             PSD[i,j,k]=flux_interpolated[i,j,k]/p_momentum[i,j,k]**2

####################################
#   We now have the boundary conditions
####################################
np.amin(Lstar_invariant)
np.max(Lstar_invariant)
np.min(first_invariant_M)
np.max(first_invariant_M)
np.max(Energy_for_p)
testy=np.reshape(First_invariant_rep,-1)
plt.plot(np.reshape(First_invariant_rep,-1),np.reshape(Energy_for_p,-1),'ro')
plt.xscale('log')
plt.yscale('log')
####################################
#   For a given value of the first two invariants (which fix the
position and the angle, letting only the Lstar, more or less the altitude)
#       - find the edge conditions of the PSD (fixed time ~ position of
the satellite at that moment + linear interpolation)
#       - simulate the evolution of the PSD in funtion of the time
####################################
Lmin=3.5
Lmax=6.0
pas_L=0.1
Lstar_bin=np.arange(Lmin,Lmax+pas_L,pas_L)










########################
########################

first_invariant_M=np.logspace(math.log(20,10), math.log(1258,10), num=19)
                           # 19 logarithmically spaced values from 20 to
1258 (MeV/G)

second_invariant_J_bins=np.logspace(-19,-15, num=15)
                           # 15 logarithmically spaced bins values from
10**-19 to 10**-15 (g(cm/s)R_E)

L=cdf['L'][:]                               # L of the database (in R_E)
- L-shell ?
print cdf['L'].attrs


B=cdf['B'][:]                               # Local field magnitude (nT)
B_G=B*10**(-5)                                      # same in Gauss


FLUX=cdf['FLUX'][:,:,:]                     # count rates of energy
channels times angle
nb_obs=len(FLUX[:,0,0])                    # observations
nb_angles=len(FLUX[0,:,0])                    # angles
nb_channels=len(FLUX[0,0,:])                    # energy channels


### First adiabatic invariant M (momentum calculated with
energy-momentum relation)
momentum=energy_MeV             # vector containing the relativistic
momentums
for k in range(0, nb_channels-1):
     ### !!!!!!!!! PROBLEME : sqrt D UN NEGATIF ??
     momentum[k]=math.sqrt(max(0,(energy_MeV[k])**2-(m_e)**2))

first_invariant=np.zeros(shape=(nb_obs,nb_angles,nb_channels))
for i in range(0, nb_obs-1):
     for j in range(0, nb_angles-1):
         for k in range(0, nb_channels-1):
first_invariant[i,j,k]=(momentum[k]*math.sin(angle_rad[j]))**2/(2*m_e*B_G[i])






# FLUX

differential_flux_per_angle=np.zeros(shape=(nb_obs,nb_angles))
for i in range(0, nb_obs-1):
     for j in range(0, nb_angles-1):
         for k in range(0, nb_channels-1):
differential_flux_per_angle[i,j]=differential_flux_per_angle[i,j]+FLUX[i,j,k]/energy[k]


