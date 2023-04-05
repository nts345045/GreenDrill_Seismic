"""
:method: Fig3_WHB_Density_Profiles.py

"""
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

sys.path.append('..')
import util.Firn_Density as fdu



ROOT = os.path.join('..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
# Wiechert-Herglotz-Bateman Reference Spread Models
UFMT = os.path.join(ROOT,'velocity_models','Spread_{SP}_v5_ele_MK2_ptO3_sutured_GeoRod_WHB_ODR_LHSn100.csv')

# Wiechert-Herglotz-Bateman Reference Model(s)
UDAT = os.path.join(ROOT,'velocity_models','Full_v5_ele_MK2_ptO3_sutured_WHB_ODR_LHSn100.csv')

flist = [('Avg',UDAT)]
for f_ in ['NS01','NS02','NS03','WE01','WE02','WE03']:
	flist.append((f_,UFMT.format(SP=f_)))



# Initialize figure
plt.figure(figsize=(7.75,6.5))
# Render Figure A
ax1 = plt.subplot(221)
# Render Figure B
ax2 = plt.subplot(223)
# Render Figure C
ax3 = plt.subplot(122)

clist = ['black','blue','dodgerblue','cornflowerblue','firebrick','red','salmon']

VN = 3850.

zmax,zmin = 0, 10000
for i_,nfv_ in enumerate(flist):
	n_,f_ = nfv_
	df = pd.read_csv(f_)
	zm = df['mean z'].values
	vm = 1e3/df['median u(z)'].values
	vu = 1e3/df['Q975 u(z)'].values
	vl = 1e3/df['Q025 u(z)'].values

	ax1.plot(vm,zm,'-',c=clist[i_],label=n_)
	ax1.fill_betweenx(zm,vl,vu,color=clist[i_],alpha=0.25)

	# Calculate Robin Densities
	pRm = fdu.rho_robin(vm)
	pRu = fdu.rho_robin(vu)
	pRl = fdu.rho_robin(vl)

	# Calculate Kohnen Densities
	pKm = fdu.rho_kohnen(vm)
	pKu = fdu.rho_kohnen(vu)
	pKl = fdu.rho_kohnen(vl)
	if n_ == 'Avg':
		ax2.plot(pRm,zm,'-',c=clist[i_],label='Robin $\\rho(z)$')
		ax2.plot(pKm,zm,'--',c=clist[i_],label='Kohnen $\\rho(z)$')
	else:
		ax2.plot(pRm,zm,'-',c=clist[i_])
		ax2.plot(pKm,zm,'--',c=clist[i_])

	ax2.fill_betweenx(zm,pKl,pKu,color=clist[i_],alpha=0.25)
	ax2.fill_betweenx(zm,pRl,pRu,color=clist[i_],alpha=0.25)


	ax3.plot(pRm,zm,'-',c=clist[i_])
	ax3.plot(pKm,zm,'--',c=clist[i_])
	ax3.fill_betweenx(zm,pKl,pKu,color=clist[i_],alpha=0.25)
	ax3.fill_betweenx(zm,pRl,pRu,color=clist[i_],alpha=0.25)

	if df['mean z'].min() < zmin:
		zmin = df['mean z'].min()
	if df['mean z'].max() > zmax:
		zmax = df['mean z'].max()




ax1.set_ylim([zmax+1,zmin-1])
ax2.set_ylim([zmax+1,zmin-1])
ax3.set_xlim([860,900])
ax3.set_ylim([80, 50])


ax1.set_xlabel('Seismic Velocity [$v(z)$] (m sec$^{-1}$)')
ax2.set_xlabel('Seismic Density [$\\rho(z)$] (kg m$^{-3}$)')
ax3.set_xlabel('Seismic Density [$\\rho(z)$] (kg m$^{-3}$)')

ax1.set_ylabel('Depth Below Glacier Surface (m BGS)')
ax2.set_ylabel('Depth Below Glacier Surface (m BGS)')
ax3.set_ylabel('Depth Below Glacier Surface (m BGS)')

ax3.grid()

ax1.legend()
ax2.legend()

ax1.text(3900,10,'a',fontweight='extra bold',fontstyle='italic',fs=16)
ax2.text(900,10,'b',fontweight='extra bold',fontstyle='italic',fs=16)
ax3.text(897.5,51,'c',fontweight='extra bold',fontstyle='italic')
plt.show()
	
	