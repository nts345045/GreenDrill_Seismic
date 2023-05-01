"""
:method: Fig4_WHB_Density_Profiles.py

"""
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

sys.path.append('..')
import util.Firn_Density as fdu



ROOT = os.path.join('..','..','..','..','..','processed_data','Hybrid_Seismic','VelCorrected_t0','Prudhoe_Dome')
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
rho_ice = 870.
print('FAC Calculations\nProfile\tQ025\tQ50\tQ975\n\tIQR/2')

zmax,zmin = 0, 10000
for i_,nfv_ in enumerate(flist):
	n_,f_ = nfv_
	df = pd.read_csv(f_)
	zm = df['mean z'].values
	zu = df['Q975 z'].values
	zl = df['Q025 z'].values
	vm = 1e3/df['median u(z)'].values
	vu = 1e3/df['Q975 u(z)'].values
	vl = 1e3/df['Q025 u(z)'].values

	ax1.plot(vm,zm,'-',c=clist[i_],label=n_)
	ax1.fill_betweenx(zm,vl,vu,color=clist[i_],alpha=0.25)
	# Calculate Robin Densities
	pRm = fdu.rho_robin(vm)
	pRu = fdu.rho_robin(vu)
	pRl = fdu.rho_robin(vl)
	FAC_Rm = fdu.calc_FAC(pRm,zm,rhoice=rho_ice)
	FAC_Ru = fdu.calc_FAC(pRu,zu,rhoice=rho_ice)
	FAC_Rl = fdu.calc_FAC(pRl,zl,rhoice=rho_ice)

	# Calculate Kohnen Densities
	pKm = fdu.rho_kohnen(vm)
	pKu = fdu.rho_kohnen(vu)
	pKl = fdu.rho_kohnen(vl)
	FAC_Km = fdu.calc_FAC(pKm,zm,rhoice=rho_ice)
	FAC_Ku = fdu.calc_FAC(pKu,zu,rhoice=rho_ice)
	FAC_Kl = fdu.calc_FAC(pKl,zl,rhoice=rho_ice)
	print(nfv_[0])
	print('Robin\t{:.4e}\t{:.4e}\t{:.4e}\t{:.4e}'.format(FAC_Rl,FAC_Rm,FAC_Ru,(FAC_Ru - FAC_Rl)/2))
	print('Kohnen\t{:.4e}\t{:.4e}\t{:.4e}\t{:.4e}'.format(FAC_Kl,FAC_Km,FAC_Ku,(FAC_Ku - FAC_Kl)/2))


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

ax2.fill_between([860,900],[50]*2,[80]*2,color='purple',alpha=0.25)
ax2.text(901,49,'c',fontweight='extra bold',fontstyle='italic')


ax1.set_xlabel('Seismic Velocity [$v(z)$] (m sec$^{-1}$)',labelpad=1)
ax2.set_xlabel('Seismic Density [$\\rho(z)$] (kg m$^{-3}$)',labelpad=1)
ax3.set_xlabel('Seismic Density [$\\rho(z)$] (kg m$^{-3}$)',labelpad=1)

ax1.set_ylabel('Depth Below Glacier Surface (m BGS)',labelpad=1)
ax2.set_ylabel('Depth Below Glacier Surface (m BGS)',labelpad=1)
ax3.set_ylabel('Depth Below Glacier Surface (m BGS)',labelpad=1)

ax1.grid()
ax2.grid()
ax3.grid()

ax1.legend()
ax2.legend()

ax1.text(3900,10,'a',fontweight='extra bold',fontstyle='italic',fontsize=16)
ax2.text(897.5,10,'b',fontweight='extra bold',fontstyle='italic',fontsize=16)
ax3.text(897.5,51.5,'c',fontweight='extra bold',fontstyle='italic',fontsize=16)
plt.show()
	
	