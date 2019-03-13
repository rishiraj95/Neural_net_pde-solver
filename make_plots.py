import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate


filename='plot_variables.pickle'
with open(filename,'rb') as f:
    plot_variables=pickle.load(f)
#plot_variables is a tuple containing (xx,yy,plot_variable,plot_data)
xx,yy,plot_variable,plot_data=plot_variables
x=np.linspace(-1,1,256)
y=np.linspace(-1,1,256)
interpfun=interpolate.interp2d(x,y,plot_variable,kind='cubic')
print(interpfun(0.6,0.8))
fig=plt.figure()
pc=plt.contourf(xx,yy,plot_variable,cmap='hot')
cb=fig.colorbar(pc,shrink=0.8)
lab=plt.clabel(pc,colors='k')
plt.show()
plt.close(plt.figure())    

