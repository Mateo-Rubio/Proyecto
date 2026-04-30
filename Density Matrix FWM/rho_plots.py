import numpy as np
import matplotlib.pyplot as plt
t = 10000
z = 10000
Delta = [10000,0]
Gamma_ba = 1 
mu_ba = 1 
h_bar = 1
c = 3e8 
lmb = np.array([852.35, 822.48, 794.39])
k = np.array([2*np.pi/lmbi for lmbi in lmb]) 
om = k*c
E0 = [1,1,1]
E = [E0[i] * np.exp(complex(0,ki*z)) for i,ki in enumerate(k)]
e = [np.exp(complex(0,-om[i]*t)) for i in range(3)]
p_ba = []
for Delta3 in np.arange(-10,10):
    p_ba.append((mu_ba/h_bar)*(
        E[0]*e[0]/complex(Delta[0],-Gamma_ba) + 
        E[1]*e[1]/complex(2*Delta[0]-Delta3,-Gamma_ba) + 
        E[2]*e[2]/complex(Delta3,-Gamma_ba)
        )
    )
fig, axis = plt.subplots()
axis.plot(np.arange(-10,10),p_ba)
axis.set(xlabel = r'$\Delta_3$', ylabel = r'$\rho_{ba}$',title= r'Coherecia $\rho_{ba}$ Parte Real')
plt.show()