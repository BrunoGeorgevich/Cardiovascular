# -*- coding: utf-8 -*-

"""
Created on Fri Jun 13 15:44:58 2019

@author: Bruno Georgevich Ferreira
"""

import numpy as np
import matplotlib.pyplot as plt

#Constantes
Emax = 2
Emin = 0.06
HR = 75

Rs = 1
Rm = 0.005
Ra = 0.001
Rc = 0.0398

Cr = 4.4
Cs = 1.33
Ca = 0.08
Ls = 0.0005
Vo = 10

Dm = 1
Da = 1

tc = 60/HR
Tmax = 0.2 + 0.15*tc

#Funções
tn = lambda t: (t % tc)/Tmax
En = lambda t: 1.55*(((tn(t)/0.7)**1.9)/(1 + ((tn(t)/0.7)**1.9)))*((1)/(1 + ((tn(t)/1.17)**21.9)))
E  = lambda t: (Emax - Emin)*En(t) + Emin

def RK4(f):
    return lambda t, y, dt: (
            lambda dy1: (
            lambda dy2: (
            lambda dy3: (
            lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
            )( dt * f( t + dt  , y + dy3   ) )
	    )( dt * f( t + dt/2, y + dy2/2 ) )
	    )( dt * f( t + dt/2, y + dy1/2 ) )
	    )( dt * f( t       , y         ) )
            
def valves(Pae, Pao, Pve):
    if Pae > Pve:
        Da = 0
        Dm = 1
    elif Pve > Pao:
        Da = 1
        Dm = 0
    else:
        Da = 0
        Dm = 0  
    return Dm,Da

A = lambda t,y: np.array([
                    [-((Dm/Rm)+(Da/Ra))*E(t), Dm/Rm, 0, Da/Ra, 0],
                    [Dm*E(t)/(Rm*Cr),-((1/Rs) + (Dm/Rm))*(1/Cr), 0,0,1/(Rs*Cr)],
                    [0,0,-Rc/Ls, 1/Ls, -1/Ls],
                    [Da*E(t)/(Ra*Ca), 0, -1/Ca, -Da/(Ra*Ca),0],
                    [0,1/(Rs*Cs),1/Cs,0,-1/(Rs*Cs)]
                 ])

f = lambda t,y: np.array([
                    [((Dm/Rm) + (Da/Ra))*E(t)*Vo],
                    [-(Dm/(Rm*Cr))*E(t)*Vo],
                    [0],
                    [-(Da/(Ra*Ca))*E(t)*Vo],
                    [0]
                ])  
    
x_dot   = lambda t,y:A(t,y)@np.array(x) + f(t,y)
Suga_Sagawa = lambda t:E(t)*(Vve - Vo)

dy = RK4(x_dot)

Vve,Pae,Qa,Pao,Ps = [ 140,   5, 0,  90,   90]

x = [[Vve],[Pae],[Qa],[Pao],[Ps]]

Vve_val = []
Pve_val = []
Pae_val = []
Pao_val = []
Qa_val = []
E_val = []
tempo = []

t, dt = 0, 0.0001
while t <= 3:     
        
    Pve = Suga_Sagawa(t)    
    Dm, Da = valves(Pae, Pao, Pve)  
    
    Vve_val.append(Vve)
    Pve_val.append(Pve)
    Pae_val.append(Pae)
    Pao_val.append(Pao)
    Qa_val.append(Qa)
    E_val.append(E(t))
    tempo.append(t)
    
    t, dx = t + dt, dy(t,x,dt)
    x += dx
    
    Vve,Pae,Qa,Pao,Ps = x.T[0]       
#%%
plt.figure(1, figsize=(16,9))
plt.subplot(3,1,1)
plt.title('Simulação Hemodinâmica de um Coração Normal')
plt.grid()
plt.ylabel('Pressões (mmHg)')
plt.plot(tempo, Pve_val, tempo, Pao_val, tempo, Pae_val)
plt.legend(['Ventrículo Esquerdo','Aórtica','Átrio Esquerdo'])
plt.subplot(3,1,2)
plt.grid()
plt.ylabel('Fluxo Aórtico (m/s)')
plt.plot(tempo, Qa_val, 'r')
plt.subplot(3,1,3)
plt.grid()
plt.ylabel('Volume no Ventrículo Esquerdo (ml)')
plt.xlabel('Tempo')
plt.plot(tempo, Vve_val, 'purple')
plt.show()
#%%