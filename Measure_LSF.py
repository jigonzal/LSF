import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.modeling import models, fitting
from astropy.io import fits
import argparse
from astropy.table import Table

try:
	import seaborn as sns
	sns.set_style("white", {'legend.frameon': True})
	sns.set_style("ticks", {'legend.frameon': True})
	sns.set_context("talk")
	sns.set_palette('Dark2',desat=1)
	cc = sns.color_palette()
except:
	cc = plt.rcParams['axes.prop_cycle'].by_key()['color']


def GetCorrelation(spectrum):
	correlation = []
	spectrum = np.nan_to_num(spectrum)
	for i in range(8):
		# i = i+1
		if i==0:
			correlation.append(np.corrcoef(spectrum, spectrum)[0][1])
		else:
			correlation.append(np.corrcoef(spectrum[:-1*i], spectrum[i:])[0][1])
	return correlation

#v0.1

parser = argparse.ArgumentParser(description="Python script that estimates the correlation of channels (i.e. LSF) in ALMA data cubes")
parser.add_argument('-Cube', type=str, required=True,help = 'Path to the Cube fits file to analyse')
parser.add_argument('-ChannelWidth', type=float, default = 0.0, required=False,help = 'Width of channels in km/s for display purposes [Default:0.0]')
args = parser.parse_args()

#Checking input arguments
print(20*'#','Checking inputs....',20*'#')
if os.path.exists(args.Cube):
    print('*** Cube',args.Cube,'found ***')
else:
    print('*** Cube',args.Cube,'not found ***\naborting..')
    exit()


CubePath = args.Cube
ChannelWidth = args.ChannelWidth
Cube = fits.open(CubePath)[0].data[0]
Mask = np.zeros_like(Cube)
FinalMask = np.mean(Mask,axis=0) 
FinalCube = np.mean(Cube,axis=0)
FinalMask[np.isnan(FinalCube)]=1
x,y = np.where(FinalMask==0)

CorrArray = []
for i in range(len(x)):
	corr = GetCorrelation(Cube[:,x[i],y[i]])
	CorrArray.append(corr)

CorrArray = np.transpose(CorrArray)
a = []
b = []
c = []

CorrelatedData = []
for i in CorrArray:
	aux = np.percentile(i,[16,50,84])
	# print(aux)
	CorrelatedData.append(aux[1])
	a.append(aux[0])
	b.append(aux[1])
	c.append(aux[2])
x = np.arange(len(a))


a = np.array(a)
b = np.array(b)
c = np.array(c)
CorrelatedData = np.array(CorrelatedData)

x1 = x*1
b1 = b*1

x = np.append(-1*x[1:][::-1],x)
a = np.append(a[1:][::-1],a)
b = np.append(b[1:][::-1],b)
c = np.append(c[1:][::-1],c)


w, h = 1.5*plt.figaspect(0.9)
fig1 = plt.figure(figsize=(w,h))

plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None, wspace=0.05, hspace=0.05)

plt.plot(x,b,color=cc[0],label='Measured Correlation',marker='o')
plt.plot(x,a,'--',color=cc[0])
plt.plot(x,c,'--',color=cc[0])
plt.fill_between(x,a,c,color=cc[0],alpha=0.5)
plt.axhline(0,color='gray')


fit_g = fitting.LevMarLSQFitter()
g_init = models.Moffat1D(amplitude=1, x_0=0, gamma=1, alpha=1)
g = fit_g(g_init, x, b)
plt.plot(np.arange(min(x),max(x),0.1),g(np.arange(min(x),max(x),0.1)),'-',label='Moffat1D fit',lw=2,color=cc[1])

b2 = np.interp(x1,x1*0.5, b1)

b2[b2<0] = 0
CorrelatedData[CorrelatedData<0] = 0

TableForOutput = Table([x1, np.around(CorrelatedData,2), np.around(b2,2)], names=('Channel', 'Correlation', 'LSF'))
TableForOutput.write(CubePath.replace('.fits','_LSF.dat'), overwrite=True,format='ascii.commented_header')


b2 = np.append(b2[1:][::-1],b2)

plt.plot(x,b2,'-',marker='o',lw=2,label='Estimated Kernel',color=cc[2])

plt.legend(loc=0,fontsize=15)
if ChannelWidth == 0.0:
	plt.xlabel('Channel')
else:
	plt.xlabel('Channel ['+str(ChannelWidth)+' km/s]')
plt.savefig(CubePath.replace('.fits','_LSF.pdf'))

