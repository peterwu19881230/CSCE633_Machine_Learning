#module import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from lmfit import Model



#%%
train=pd.DataFrame([[0,1],[2,4],[3,9],[5,16]])
test=pd.DataFrame([[1,3],[4,12]])

x=train.iloc[:,0].values
y=train.iloc[:,1].values




#%%define the model
degree=[0,1,2,3,4]
color_for_degree=["purple","blue","cyan","green","orange"]
coef_name=['a0','a1','a2','a3','a4']

def func(x,a0,a1,a2,a3,a4):
    return a0 + a1*x + a2*x ** 2 + a3*x**3 + a4*x**4 


#fit the model -> plot

#%% d=0
d=0 #param
pmodel = Model(func)
params = pmodel.make_params(a0=1,a1=0,a2=0,a3=0,a4=0) #param

for i in [4,3,2,1,0]: #param
        params[coef_name[i]].vary = False


result = pmodel.fit(y, params, x=x)
    
print(result.fit_report())
    
xnew = np.linspace(x[0], x[-1], 1000)
ynew = result.eval(x=xnew)
    
plt.ylim(bottom=0, top=20)
plt.plot(x, y, 'bo')
plt.plot(xnew, ynew, 'r-',color=color_for_degree[d])  
plt.show()



#%%  d=1
d=1 #param
pmodel = Model(func)
params = pmodel.make_params(a0=1,a1=1,a2=0,a3=0,a4=0) #param

for i in [4,3,2,1]: #param
        params[coef_name[i]].vary = False


result = pmodel.fit(y, params, x=x)
    
print(result.fit_report())
    
xnew = np.linspace(x[0], x[-1], 1000)
ynew = result.eval(x=xnew)
    
plt.ylim(bottom=0, top=20)
plt.plot(x, y, 'bo')
plt.plot(xnew, ynew, 'r-',color=color_for_degree[d])  
plt.show()
    

#%%  d=2
d=2 #param
pmodel = Model(func)
params = pmodel.make_params(a0=1,a1=1,a2=1,a3=0,a4=0) #param

for i in [4,3,2]: #param
        params[coef_name[i]].vary = False


result = pmodel.fit(y, params, x=x)
    
print(result.fit_report())
    
xnew = np.linspace(x[0], x[-1], 1000)
ynew = result.eval(x=xnew)
    
plt.ylim(bottom=0, top=20)
plt.plot(x, y, 'bo')
plt.plot(xnew, ynew, 'r-',color=color_for_degree[d])  
plt.show()

    
#%%  d=3
d=3 #param
pmodel = Model(func)
params = pmodel.make_params(a0=1,a1=1,a2=1,a3=1,a4=0) #param

for i in [4,3]: #param
        params[coef_name[i]].vary = False


result = pmodel.fit(y, params, x=x)
    
print(result.fit_report())
    
xnew = np.linspace(x[0], x[-1], 1000)
ynew = result.eval(x=xnew)
    
plt.ylim(bottom=0, top=20)
plt.plot(x, y, 'bo')
plt.plot(xnew, ynew, 'r-',color=color_for_degree[d])  
plt.show()

    
#%%  d=4
d=4 #param
pmodel = Model(func)
params = pmodel.make_params(a0=1,a1=1,a2=1,a3=1,a4=1) #param

for i in [4]: #param
        params[coef_name[i]].vary = False


result = pmodel.fit(y, params, x=x)
    
print(result.fit_report())
    
xnew = np.linspace(x[0], x[-1], 1000)
ynew = result.eval(x=xnew)
    
plt.ylim(bottom=0, top=20)
plt.plot(x, y, 'bo')
plt.plot(xnew, ynew, 'r-',color=color_for_degree[d])  
plt.show()

    
    
    
    
     
    
    