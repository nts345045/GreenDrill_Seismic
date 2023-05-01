"""
Fig_S1_Shot_Gather_Geometry.py
"""

import matplotlib.pyplot as plt
import numpy as np

# Set source location as 0
sxv = 0
# Receiver Locations (drop every 4th design)
rxv = [x for x in np.arange(0,450,10) if x%40!=0 ]
rxv = np.array(rxv)
# Shift back so furthest back receiver is at 0 locally
rxv -= 10

# Shift forward to have source-receiver offset
rxv += 480
# for r_ in rxv
# Surface Grid discretization
xv = np.arange(-30,480*3 + 1,1)

# Model for surface topography
def surfmod(xv,A=2,T=200,b=1300):
	return A*np.sin((2*np.pi/T)*(xv)) + b
# Model for bed topography
def bedmod(xv,A=10,T=500,m=-0.05,b=800):
	return A*np.sin((2*np.pi/T)*(xv)) + m*xv + b

# Formatting objects
c_list_reds = ['maroon','firebrick','red','orangered','darkorange']
c_list_blues = ['midnightblue','navy','mediumblue','mediumslateblue','dodgerblue']

# Create surface elevations
surfH = surfmod(xv)
recH = surfmod(rxv)
bedH = bedmod(xv,m=0.05)
srcH = surfmod(sxv)





plt.figure(figsize=(7.5,8))
plt.subplot(211)
plt.fill_between(xv,surfH,bedH,color='dodgerblue',alpha=0.25)
plt.fill_between(xv,bedH,np.ones(len(xv))*700,color='black',alpha=0.25)
plt.plot(xv,surfH,color='dodgerblue')#,label='Ice surface')
plt.plot(xv,bedH,color='black')#,label='Ice-bed interface')
plt.plot(rxv,recH,'kv',label='Geophones')
plt.plot(sxv,srcH,'*',ms=16,label='Shot',mfc='red',mec='red')

bed_pts = []
for i_,r_ in enumerate(rxv):
	path_x = np.array([sxv,np.mean([sxv,r_]),r_])
	path_y = np.array([surfmod(sxv),bedmod(np.mean([sxv,r_]),m=0.05),surfmod(r_)])

	bed_pts.append([np.mean([sxv,r_]),bedmod(np.mean([sxv,r_]),m=0.05)])
	if i_ == 0:
		hp = plt.plot(path_x,path_y,'r-',linewidth=0.5,label='Reflected raypath')
	else:
		plt.plot(path_x,path_y,'r-',linewidth=0.5)

bed_pts = np.array(bed_pts)

plt.errorbar(np.mean(bed_pts[:,0]),np.mean(bed_pts[:,1]),\
		     xerr=[[np.mean(bed_pts[:,0]) - np.min(bed_pts[:,0])],\
		     	   [np.max(bed_pts[:,0]) - np.mean(bed_pts[:,0])]],\
			 yerr=[[np.mean(bed_pts[:,1]) - np.min(bed_pts[:,1] - 5)],\
		     	   [5 + np.max(bed_pts[:,1]) - np.mean(bed_pts[:,1])]],\
	     	 capsize=5,fmt='o',color='red',label='$H_{bed}$ estimate')


## ANNOTATE ##
txt_fmt = {'fontweight':'extra bold','fontsize':14}
plt.text(np.mean(bed_pts[:,0]),1200,'Ice',color='dodgerblue',**txt_fmt)
plt.text(150,730,'Rock',color='black',**txt_fmt)
plt.text(np.mean(bed_pts[:,0]),750,'Illuminated\nregion',color='red',**txt_fmt,ha='center',va='center')
## WINDOW DRESSING ##
plt.ylabel('Elevation relative to sea level [m ASL]')
plt.xlabel('Source-receiver offset [m]')
plt.xlim([rxv.max() + 30,sxv - 30])
plt.ylim([700,1350])
plt.text(900,1160,'a',color='black',fontweight='extra bold',fontstyle='italic',fontsize=16)


plt.legend(loc='lower left')
plt.plot(rxv,recH,'kv')


plt.subplot(212)
bedH = bedmod(xv)



plt.fill_between(xv,surfH,bedH,color='dodgerblue',alpha=0.25)
plt.fill_between(xv,bedH,np.ones(len(xv))*600,color='black',alpha=0.25)
plt.plot(xv,surfH,color='dodgerblue')
plt.plot(xv,bedH,color='black')

rxv = [x for x in np.arange(0,450,10) if x%40!=0 ]
rxv = np.array(rxv)



for I_,SX_ in enumerate(np.arange(0,5)*240 + 240):

	plt.plot(SX_,surfmod(SX_),'*',ms=16,label='Shot %d'%(I_ + 1),mfc=c_list_reds[I_],mec='red')

	bed_pts = []
	for i_,r_ in enumerate(rxv):
		path_x = np.array([SX_,np.mean([SX_,r_]),r_])
		path_y = np.array([surfmod(SX_),bedmod(np.mean([SX_,r_])),surfmod(r_)])

		bed_pts.append([np.mean([SX_,r_]),bedmod(np.mean([SX_,r_]))])
		if i_ == 0:
			hp = plt.plot(path_x,path_y,'-',color=c_list_reds[I_],linewidth=0.5)
		else:
			plt.plot(path_x,path_y,'-',color=c_list_reds[I_],linewidth=0.5)

	bed_pts = np.array(bed_pts)

	plt.errorbar(np.mean(bed_pts[:,0]),np.mean(bed_pts[:,1]),\
			     xerr=[[np.mean(bed_pts[:,0]) - np.min(bed_pts[:,0])],\
			     	   [np.max(bed_pts[:,0]) - np.mean(bed_pts[:,0])]],\
				 yerr=[[np.mean(bed_pts[:,1]) - np.min(bed_pts[:,1] - 5)],\
			     	   [5 + np.max(bed_pts[:,1]) - np.mean(bed_pts[:,1])]],\
		     	 capsize=5,fmt='o',color=c_list_reds[I_])#,label='Shot #%d\n$H_{bed}$ estimate'%(I_ +1))

plt.xlim([rxv.min() - 30,SX_ + 130])
plt.ylim([600,1350])

plt.text(0,700,'b',color='black',fontweight='extra bold',fontstyle='italic',fontsize=16)

plt.legend(ncol=1,loc='lower right',fontsize=10)
plt.plot(rxv,recH,'kv')
plt.text(500,690,'Overlapping illuminated\nregions',color=c_list_reds[2],ha='center',va='center',**txt_fmt)
plt.xlabel('Distance along spread [m]')
plt.ylabel('Elevation relative to sea level [m ASL]')










#### CREATE ZOOMED VIEWS


# Create surface elevations
surfH = surfmod(xv)
recH = surfmod(rxv)
bedH = bedmod(xv,m=0.05)
srcH = surfmod(sxv)

plt.figure(figsize=(7.5,8))
plt.subplot(211)
plt.fill_between(xv,surfH,bedH,color='dodgerblue',alpha=0.25)
plt.fill_between(xv,bedH,np.ones(len(xv))*700,color='black',alpha=0.25)
# plt.plot(xv,surfH,color='dodgerblue',label='Ice surface')
plt.plot(xv,bedH,color='black',label='Ice-bed interface')
# plt.plot(rxv,recH,'kv',label='Geophones')
# plt.plot(sxv,srcH,'*',ms=16,label='Shot',mfc='red',mec='red')

bed_pts = []
for i_,r_ in enumerate(rxv):
	path_x = np.array([sxv,np.mean([sxv,r_]),r_])
	path_y = np.array([surfmod(sxv),bedmod(np.mean([sxv,r_]),m=0.05),surfmod(r_)])

	bed_pts.append([np.mean([sxv,r_]),bedmod(np.mean([sxv,r_]),m=0.05)])
	if i_ == 0:
		hp = plt.plot(path_x,path_y,'r-',linewidth=0.5,label='Reflected raypath')
	else:
		plt.plot(path_x,path_y,'r-',linewidth=0.5)

bed_pts = np.array(bed_pts)

plt.errorbar(np.mean(bed_pts[:,0]),np.mean(bed_pts[:,1]),\
		     xerr=[[np.mean(bed_pts[:,0]) - np.min(bed_pts[:,0])],\
		     	   [np.max(bed_pts[:,0]) - np.mean(bed_pts[:,0])]],\
			 yerr=[[np.mean(bed_pts[:,1]) - np.min(bed_pts[:,1] - 5)],\
		     	   [5 + np.max(bed_pts[:,1]) - np.mean(bed_pts[:,1])]],\
	     	 capsize=5,fmt='o',color='red',label='$H_{bed}$ estimate')


## ANNOTATE ##
txt_fmt = {'fontweight':'extra bold','fontsize':14}
plt.text(400,875,'Ice',color='dodgerblue',**txt_fmt)
plt.text(300,760,'Rock',color='black',**txt_fmt)
plt.text(np.mean(bed_pts[:,0]),775,'|- Illuminated region -|',color='red',**txt_fmt,ha='center',va='center')
plt.text(np.mean(bed_pts[:,0])+50,805,'^\n$H_{bed}$ 95% CI\nv',color='red',va='center',ha='center')
## WINDOW DRESSING ##
plt.ylabel('Elevation relative to sea level [m ASL]',labelpad=15)
plt.xlabel('Source-receiver offset [m]')
plt.xlim([rxv.max() + 30,sxv - 30])
plt.ylim([750,900])

plt.text(450,880,'a',color='black',fontweight='extra bold',fontstyle='italic',fontsize=16)

plt.legend(loc='lower left')
plt.plot(rxv,recH,'kv')





plt.subplot(212)
bedH = bedmod(xv)



plt.fill_between(xv,surfH,bedH,color='dodgerblue',alpha=0.25)
plt.fill_between(xv,bedH,np.ones(len(xv))*600,color='black',alpha=0.25)
plt.plot(xv,surfH,color='dodgerblue')
plt.plot(xv,bedH,color='black')

rxv = [x for x in np.arange(0,450,10) if x%40!=0 ]
rxv = np.array(rxv)



for I_,SX_ in enumerate(np.arange(0,5)*240 + 240):

	# plt.plot(SX_,surfmod(SX_),'*',ms=16,label='Shot %d'%(I_ + 1),mfc=c_list_reds[I_],mec='red')

	bed_pts = []
	for i_,r_ in enumerate(rxv):
		path_x = np.array([SX_,np.mean([SX_,r_]),r_])
		path_y = np.array([surfmod(SX_),bedmod(np.mean([SX_,r_])),surfmod(r_)])

		bed_pts.append([np.mean([SX_,r_]),bedmod(np.mean([SX_,r_]))])
		if i_ == 0:
			hp = plt.plot(path_x,path_y,'-',color=c_list_reds[I_],linewidth=0.5)
		else:
			plt.plot(path_x,path_y,'-',color=c_list_reds[I_],linewidth=0.5)

	bed_pts = np.array(bed_pts)

	plt.errorbar(np.mean(bed_pts[:,0]),np.mean(bed_pts[:,1]),\
			     xerr=[[np.mean(bed_pts[:,0]) - np.min(bed_pts[:,0])],\
			     	   [np.max(bed_pts[:,0]) - np.mean(bed_pts[:,0])]],\
				 yerr=[[np.mean(bed_pts[:,1]) - np.min(bed_pts[:,1] - 5)],\
			     	   [5 + np.max(bed_pts[:,1]) - np.mean(bed_pts[:,1])]],\
		     	 capsize=5,fmt='o',color=c_list_reds[I_],label='Shot #%d\n$H_{bed}$ estimate'%(I_ +1))

plt.xlim([rxv.min() - 30,SX_ + 130])
plt.ylim([700,900])

plt.text(0,875,'b',color='black',fontweight='extra bold',fontstyle='italic',fontsize=16)


plt.legend(ncol=1,loc='lower right',fontsize=10)
plt.plot(rxv,recH,'kv')
plt.text(500,730,'Overlapping illuminated\nregions',color=c_list_reds[2],ha='center',va='center',**txt_fmt)
plt.xlabel('Distance along spread [m]')
plt.ylabel('Elevation relative to sea level [m ASL]',labelpad=15)



plt.show()