#!/usr/bin/env python
# coding: utf-8

# # Cálculo para las tres primeras capas

# ## Constantes

# In[4]:


#Definimos las constantes necesarias

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MTot = 5  #masa total
LTot = 40 #luminosidd total
RTot = 12.0 #radio total
RIni = 0.9*RTot #radio inicial
num_capas = 101 #número de capas
X = 0.8 #contenido de hidrogeno
Y = 0.16 #contenido de helio
Z = 1-Y-X #elementos pesados
mu = 1/((2*X)+(3*Y/4)+(Z/2)) #peso molecular medio

A1 = 1.9022*mu*MTot #cte para la temp de las primeras capas
A2 = 10.645*np.sqrt(MTot/(mu*Z*LTot*(1+X))) #cte para la pres de las primeras capas
Cp = 8.084*mu #cte para calcular f_i de las pres
Ct = 0.01679*Z*(1+X)*mu**2 #cte para calcular f_i de la temp


h = -RIni/(num_capas-1) #paso de integración
ERelMax = 0.0001
X1_PP = X
X2_PP = X
X1_CN = X
X2_CN = Z / 3
rho = MTot / (4/3) * np.pi * RTot**3

T=np.zeros(num_capas)
Test=np.zeros(num_capas)
Tcal=np.zeros(num_capas)

P=np.zeros(num_capas)
Pest=np.zeros(num_capas)
Pcal=np.zeros(num_capas)

fp=np.zeros(num_capas)
ft=np.zeros(num_capas)
fm=np.zeros(num_capas)
fl=np.zeros(num_capas)


M=np.zeros(num_capas)
Mcal=np.zeros(num_capas)
Mest=np.zeros(num_capas)

L=np.zeros(num_capas)
Lcal=np.zeros(num_capas)
GenEnergiaPP=np.zeros(num_capas)
GenEnergiaCN=np.zeros(num_capas)

r = np.zeros(num_capas) #arrays vacios para almacenar los datos
n=np.zeros(num_capas)

novotny = np.ones(num_capas) * 2.5


# In[5]:


DEBUG = False
#DEBUG = True


# ## Definiciones

# In[6]:


def T_primeras_capas(A1, r, RTot):
    return A1*((1/r)-(1/RTot))

def P_primeras_capas(A2, T):
    return A2*T**(4.25)

def f_i_presion(Cp, P, T, MTot, r):
    return (-Cp*P*MTot)/(T*r**2)

def f_i_temperatura(Ct, P, LTot, T, r):
    return (-Ct*P**2*LTot)/(T**8.5*r**2)


# ## Bucle para las tres primeras capas

# In[7]:


def Tres_primeras_capas(i):
    r[i]=RIni+(i*h)
    T[i]=T_primeras_capas(A1, r[i], RTot)
    P[i]=P_primeras_capas(A2, T[i])
    L[i]=LTot
    M[i]=MTot
    fp[i]=f_i_presion(Cp, P[i], T[i], MTot, r[i])
    ft[i]=f_i_temperatura(Ct, P[i], LTot, T[i], r[i])
    fm[i]=0.0
    fl[i]=0.0
    
    if DEBUG:
        print(f'i: {i}, fp[i]: {fp[i]}')
        print(f'i: {i}, ft[i]: {ft[i]}')
        print(f'i: {i}, fl[i]: {fl[i]}')
        print(f'i: {i}, fm[i]: {fm[i]}')  


# In[8]:


i=-1

while i<=1:
    i += 1
    Tres_primeras_capas(i)
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
    
# Ya tenemos las tres primeras capas.              


# ## Integración envoltura radiativa

# ## 1 y 2.Definiciones de presión y temperatura y calculo del radio
# 

# In[9]:


def step_2(i):
    delta1_P = h * fp[i] - h * fp[i-1]
    delta2_P = h * fp[i] - 2 * h * fp[i-1] + h * fp[i-2]
    delta1_T = h * ft[i] - h * ft[i-1]
    
    Pest[i+1] = P[i] + h * fp[i] + 0.5 * delta1_P + 5 / 12 * delta2_P
    Test[i+1] = T[i] + h * ft[i] + 0.5 * delta1_T
    if DEBUG:
        print(f'step_2> delta1_P, delta2_P: {delta1_P}, {delta2_P}')
        print(f'step_2> Pest: {Pest[i+1]}')
        print(f'step_2> delta1_T: {delta1_T}')
        print(f'step_2> Test: {Test[i+1]}')


def step_3(i):
    fm[i+1] = 0.01523 * mu * (Pest[i+1] / Test[i+1]) * r[i+1]**2
    delta1_M = h * fm[i+1] - h * fm[i]
    Mcal[i+1] = M[i] + h * fm[i+1] - 0.5 * delta1_M
    if DEBUG:
        print(f'step_3> fm[i+1], fm[i]: {fm[i+1]}, {fm[i]}')
        print(f'step_3> employed P, T, r: {Pest[i+1]}, {Test[i+1]}, {r[i+1]}')
        print(f'step_3> delta1_M: {delta1_M}')
        print(f'step_3> Mcal[i+1]: {Mcal[i+1]}')
    

def step_4(i):
    fp[i+1] = -8.084 * mu * (Pest[i+1] / Test[i+1]) * (Mcal[i+1] / r[i+1]**2)
    deltai_P = h * fp[i+1] - h * fp[i] #delta P i+1
    Pcal[i+1] = P[i] + h * fp[i+1] - 0.5 * deltai_P
    if DEBUG:
        print(f'step_4> fp[i+1]: {fp[i+1]}')
        print(f'step_4> delta1_P: {deltai_P}')
        print(f'step_4> Pcal[i+1]: {Pcal[i+1]}')
    

def step_5(i):
    return abs(Pcal[i+1] - Pest[i+1]) / Pcal[i+1]
    
        
def GenEnergia(T):
    """T is given in units of 10**7 K"""
    # cycle PP
    TT = T*10 
    #print(TT)
    if 4 < TT <= 6:
        E1_PP = 10 ** -6.84
        nu_PP = 6
    elif  TT <= 9.5:
        E1_PP = 10 ** -6.04
        nu_PP = 5
    elif  TT <= 12:
        E1_PP = 10 ** -5.56
        nu_PP = 4.5
    elif  TT <= 16.5:
        E1_PP = 10 ** -5.02
        nu_PP = 4
    elif  TT <= 24:
        E1_PP = 10 ** -4.4
        nu_PP = 3.5
    else:
        E1_PP = 0
        nu_PP = 0

    # cycle CN
    if 12 < TT <= 16:
        E1_CN = 10 ** -22.2
        nu_CN = 20
    elif  TT <= 22.5:
        E1_CN = 10 ** -19.8
        nu_CN = 18
    elif  TT <= 27.5:
        E1_CN = 10 ** -17.1
        nu_CN = 16
    elif  TT <= 36:
        E1_CN = 10 ** -15.6
        nu_CN = 15
    elif TT <= 50:
        E1_CN = 10 ** -12.5
        nu_CN = 13
    else:
        E1_CN = 0
        nu_CN = 0

    # compute energy without the density (not necessary for the comparison)
    energy_PP = E1_PP * X1_PP * X2_PP * TT ** nu_PP
    energy_CN = E1_CN * X1_CN * X2_CN * TT ** nu_CN

    if energy_PP > energy_CN:
        return E1_PP, nu_PP, X1_PP, X2_PP
    else:
        return E1_CN, nu_CN, X1_CN, X2_CN

    
# quitar parámetros que son conocidos en el ámbito del módulo (variables definidas arriba del todo)

def step_6(i):
    E1, nu, X1, X2 = GenEnergia(Test[i+1])
    fl[i+1] = 0.01845 * E1 * X1 * X2 * 10**nu * mu**2 * Pcal[i+1]**2 * Test[i+1]**(nu-2) * r[i+1]**2
    delta1_L = h * fl[i+1] - h * fl[i]
    delta2_L = h * fl[i+1] - 2 * h * fl[i] + h * fl[i-1]
    Lcal[i+1] = L[i] + h * fl[i+1] - 0.5 * delta1_L - 1/12 * delta2_L
    if DEBUG:
        print(f'step_6> E1, nu, X1, X2: {E1}, {nu}, {X1}, {X2}')
        print(f'step_6> fl[i+1]: {fl[i+1]}')
        print(f'step_6> delta1_L: {delta1_L}')
        print(f'step_6> delta2_L: {delta2_L}')
        print(f'step_6> Lcal[i+1]: {Lcal[i+1]}')
    
    
def step_7(i):
    ft[i+1] = -0.01679 * Z * (1 + X) * mu**2 * (Pcal[i+1]**2 / Test[i+1]**8.5) * (Lcal[i+1] / r[i+1]**2)
    deltai_T = h * ft[i+1] - h * ft[i]
    Tcal[i+1] = T[i] + h * ft[i+1] - 0.5 * deltai_T
    if DEBUG:
        print(f'step_7> ft[i+1]: {ft[i+1]}')
        print(f'step_7> deltai_T: {deltai_T}')
        print(f'step_7> Tcal[i+1]: {Tcal[i+1]}')
    
    
def step_8(i):
    return abs(Tcal[i+1] - Test[i+1]) / Tcal[i+1]


def step_9(i):
    

    n[i+1] =   (Tcal[i+1] / Pcal [i+1]) * ((h * fp[i+1]) / (h * ft[i+1]))


def step_10(i):
    if n  <= 2.5:
        i = i + 1


# In[10]:


#
# Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
#

i = 2  # número de la última capa calculada

Counter = []

print("-" * 78)
print("Algoritmo A.1.")
print("-" * 78)

# Utilizamos las variables lógicas loop1, loop2 y loop3
# para controlar cada uno de los bucle necesarios para
# programar el algoritmo de la fase A.1. (envoltura radiativa)

loop1 = True
while loop1:
    #print("\n*Calculando capa número:", i+1, "\n")
    #print("Ejecutando paso 1")
    r[i+1] = RIni+ (i+1)*h
    #print("Ejecutando paso 2")
    step_2(i)    
    loop2 = True
    while loop2:
        loop3 = True
        while loop3:
            #print("Ejecutando paso 3")
            step_3(i)
            #print("Ejecutando paso 4")
            step_4(i)
            
            #print(Pest[i+1], Pcal[i+1])
            diference = step_5(i)
            if diference < ERelMax:
                loop3 = False
            else:
                #print("Hacemos Pest = Pcal e iteramos loop3")
                Pest[i+1] = Pcal[i+1]
                #input('Press <CR> to continue...')
            
        #print("Ejecutando paso 6")
        
        step_6(i)
        
        #print("Ejecutando paso 7")
        step_7(i)
        
        diference = step_8(i)
        if diference < ERelMax:
            loop2 = False
        else:
            #print("Hacemos Tcal = Test e iteramos loop2")
            Test[i+1] = Tcal[i+1]
            #input('Press <CR> to continue...')

    #print("Ejecutando paso 9")
    step_9(i)
    
    decision = n[i+1]  
    if decision <= 2.5:
        loop1 = False
    else:
        # almacenamos los valores calculados y los mostramos
        P[i+1] = Pcal[i+1]
        T[i+1] = Tcal[i+1]
        L[i+1] = Lcal[i+1]
        M[i+1] = Mcal[i+1]
        print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}')   

        # pasamos a la siguiente capa
        
        i += 1
        
        Counter.append(i)
        
    
    #input('Press <CR> to continue...')


print("-" * 78)
print("Pasamos a la fase convectiva, algoritmo A.2.")
print("-" * 78)


# In[11]:


print(f'{i+1:3d} {r[i+1]:8.5f} {Pest[i+1]:10.7f} {Test[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}')


# In[12]:



last_layer = len(Counter) + 2
next_layer = len(Counter) + 3
x = r[last_layer]


# ## Nulceo convectivo

# In[13]:


k = Pest[i+1] / (Test[i+1])**2.5
print(k)

def step_2_bis(i):
    delta1_T = h * ft[i] - h * ft[i-1]
    Test[i+1] = T[i] + h * ft[i] + 0.5 * delta1_T
    if DEBUG:
        print(f'step_2> ft[i],ft[i-1],ft[i+1]: {ft[i]},{ft[i-1]},{ft[i+1]}')
        print(f'step_2> delta1_T: {delta1_T}')
        print(f'step_2> Test: {Test[i+1]}')
    
def politropo_1(i):
    Pest[i+1] = k * Test[i+1] ** 2.5
    if DEBUG:
        print(f'politropo_1> Pest: {Pest[i+1]}')
        
        
def step_6_bis(i):
    E1, nu, X1, X2 = GenEnergia(Tcal[i+1])
    fl[i+1] = 0.01845 * E1 * X1 * X2 * 10**nu * mu**2 * Pcal[i+1]**2 * Tcal[i+1]**(nu-2) * r[i+1]**2
    delta1_L = h * fl[i+1] - h * fl[i]
    delta2_L = h * fl[i+1] - 2 * h * fl[i] + h * fl[i-1]
    Lcal[i+1] = L[i] + h * fl[i+1] - 0.5 * delta1_L - 1/12 * delta2_L
    if DEBUG:
        print(f'step_6> E1, nu, X1, X2: {E1}, {nu}, {X1}, {X2}')
        print(f'step_6> fl[i+1]: {fl[i+1]}')
        print(f'step_6> delta1_L: {delta1_L}')
        print(f'step_6> delta2_L: {delta2_L}')
        print(f'step_6> Lcal[i+1]: {Lcal[i+1]}')
        
def step_7_bis(i):
    if r[i+1] == 0:
        Tcal[i+1] = Test[i+1]
        
    else:
        ft[i+1] = -3.234 * mu * Mcal[i+1] / r[i+1]**2
        deltai_T = h * ft[i+1] - h * ft[i]
        Tcal[i+1] = T[i] + h * ft[i+1] - 0.5 * deltai_T
        
        
        if DEBUG:
            print(f'step_7> delta1_T: {deltai_T}')
            print(f'step_7> Tcal: {Tcal[i+1]}')
        
def politropo_2(i):
    Pcal[i+1] = k * Tcal[i+1] ** 2.5
    if DEBUG:
        print(f'politropo_2> Pcal: {Pcal[i+1]}')
        
def step_9_bis(i):
    
    if fp[i+1]==0:
        return 0.0
    else:
        n[i+1] =   (Tcal[i+1] / Pcal [i+1]) * ((h * fp[i+1]) / (h * ft[i+1]))
    
    
    
    
    

        


# In[14]:


#
# Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
#

i = last_layer  # número de la última capa calculada

print("-" * 78)
print("Algoritmo A.2.")
print("-" * 78)

# Utilizamos las variables lógicas loop1, loop2 y loop3
# para controlar cada uno de los bucle necesarios para
# programar el algoritmo de la fase A.1. (envoltura radiativa)

loop1 = True
while loop1:
    #print("\n*Calculando capa número:", i+1, "\n")
    #print("Ejecutando paso 1")
    r[i+1] = RIni+(i+1) * h
    step_9_bis(i)
    #print("Ejecutando paso 2 bis")
    step_2_bis(i)
    
    loop2 = True
    while loop2:
        #print("Ejecutando polítropo 1")

        politropo_1(i)
            
        #print("Ejecutando paso 3")
        step_3(i)
            
            
        #print("Ejecutando paso 7 bis")
        
        step_7_bis(i)
        diference = step_8(i)
        if diference < ERelMax:
            loop2 = False
        else:
            #print("Hacemos Tcal = Test e iteramos loop2")
            Test[i+1] = Tcal[i+1]
    #print("Ejecutando polítropo 2")
    politropo_2(i)
    
    #print("Ejecutando paso 6")
    step_6_bis(i)
    
    
    decision = r[i+1]
    
    if decision <= 0:
        loop1 = False
    else:
        P[i+1] = Pcal[i+1]
        T[i+1] = Tcal[i+1]
        L[i+1] = Lcal[i+1]
        M[i+1] = Mcal[i+1]
        
        print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {n[i+1]:10.6f}') 
        # pasamos a la siguiente capa
        i += 1

print("-" * 78)
print("Hemos llegado al nucleo de la estrella.")
print("-" * 78)


# ## Interpolación 
# 

# In[15]:


# Empleamos la fórmula y=((x-x1)/(x2-x1)*(y2-y1))+ y1


# In[16]:


r_int_d = ((2.5 - n[82]) / (n[81] - n[82]) * (r[81] - r[82])) + r[82]
P_int_d = ((2.5 - n[82]) / (n[81] - n[82]) * (P[81] - P[82])) + P[82]
T_int_d = ((2.5 - n[82]) / (n[81] - n[82]) * (T[81] - T[82])) + T[82]
L_int_d = ((2.5 - n[82]) / (n[81] - n[82]) * (L[81] - L[82])) + L[82]
M_int_d = ((2.5 - n[82]) / (n[81] - n[82]) * (M[81] - M[82])) + M[82]
print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')


# In[17]:


r_int_d = ((2.5 - n[last_layer]) / (n[next_layer] - n[last_layer]) * (r[next_layer] - r[last_layer])) + r[last_layer]
P_int_d = ((2.5 - n[last_layer]) / (n[next_layer] - n[last_layer]) * (P[next_layer] - P[last_layer])) + P[last_layer]
T_int_d = ((2.5 - n[last_layer]) / (n[next_layer] - n[last_layer]) * (T[next_layer] - T[last_layer])) + T[last_layer]
L_int_d = ((2.5 - n[last_layer]) / (n[next_layer] - n[last_layer]) * (L[next_layer] - L[last_layer])) + L[last_layer]
M_int_d = ((2.5 - n[last_layer]) / (n[next_layer] - n[last_layer]) * (M[next_layer] - M[last_layer])) + M[last_layer]
print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')


# # Integración desde el centro

# ## Primeras tres capas

# In[18]:


RIni = 0.0
h = -h #paso de integración
Tc = 1.5 #Temperatura en el centro

T=np.zeros(num_capas)
Test=np.zeros(num_capas)
Tcal=np.zeros(num_capas)

P=np.zeros(num_capas)
Pest=np.zeros(num_capas)
Pcal=np.zeros(num_capas)

fp=np.zeros(num_capas)
ft=np.zeros(num_capas)
fm=np.zeros(num_capas)
fl=np.zeros(num_capas)


M=np.zeros(num_capas)
Mcal=np.zeros(num_capas)
Mest=np.zeros(num_capas)

L=np.zeros(num_capas)
Lcal=np.zeros(num_capas)
GenEnergiaPP=np.zeros(num_capas)
GenEnergiaCN=np.zeros(num_capas)

r = np.zeros(num_capas) #arrays vacios para almacenar los datos


novotny = np.ones(num_capas) * 2.5


# In[19]:


def M_tres_primeras_capas(Tc, mu, k,r):
    return 0.005077 * mu * k * np.power(Tc,1.5) *np.power(r,3)

def L_tres_primeras_capas(mu,k,Tc,r):
    E1, nu, X1, X2 = GenEnergia(Tc)
    return 0.00615 * E1 * X1 * X2 * 10**nu * mu**2 * k**2 * Tc**(3+nu) * r**3

def T_conveccion(Tc, mu, k,r):
    return Tc-0.008207 * mu**2 * k * Tc**1.5 * r**2

def P_tres_primeras_capas(k, T):
    return k * T**2.5

def fi_m_cen(k,T,r):
    Cm = 0.01523 * mu
    return Cm * k * T**1.5 * r**2

def fi_l_cen(k,T,r,mu):
    E1, nu, X1, X2 = GenEnergia(T)
    Cl_cen = 0.01845 * E1 * X1 * X2 * 10**nu * mu**2
    return Cl_cen * k**2 * T**(3+nu) * r**2

def fi_t_cen(M,r,mu):
    if r==0:
        return 0
    else:
        Ct_cen = 3.234 * mu
        return -Ct_cen * M / r**2

    
def fi_p_cen(k,T,M,r):
    if r==0:
        return 0 
    else:
        Cp = 8.084 * mu
        return -Cp * k * T**1.5 * M / r**2

    
def T_primeras_capas(A1, r, RTot):
    return A1*((1/r)-(1/RTot))


# In[20]:


def Tres_primeras_capas_cen(i):
    r[i]=RIni+(i*h)
    M[i]=M_tres_primeras_capas(Tc, mu,k,r[i])
    L[i]=L_tres_primeras_capas(mu,k,Tc,r[i])
    T[i]=T_conveccion(Tc, mu, k, r[i])
    P[i]=P_tres_primeras_capas(k, T[i])
    fm[i]=fi_m_cen(k,T[i],r[i])
    fl[i]=fi_l_cen(k,T[i],r[i],mu)
    ft[i]=fi_t_cen(M[i],r[i],mu)
    fp[i]=fi_p_cen(k,T[i],M[i],r[i])
    
    
    


# In[21]:


i=-1

while i<=1:
    i += 1
    Tres_primeras_capas_cen(i)
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
    #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')


# ## Capas posteriores

# In[22]:


#
# Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
#

i = 2  # número de la última capa calculada
Counter = []
print("-" * 78)
print("Algoritmo A.2.")
print("-" * 78)

# Utilizamos las variables lógicas loop1, loop2 y loop3
# para controlar cada uno de los bucle necesarios para
# programar el algoritmo de la fase A.1. (envoltura radiativa)

loop1 = True
while loop1:
    #print("\n*Calculando capa número:", i+1, "\n")
    #print("Ejecutando paso 1")
    r[i+1] = RIni+(i+1) * h
    #print("Ejecutando paso 2 bis")
    step_2_bis(i)
    
    loop2 = True
    while loop2:
        #print("Ejecutando polítropo 1")

        politropo_1(i)
            
        #print("Ejecutando paso 3")
        step_3(i)
            
            
        #print("Ejecutando paso 7 bis")
        
        step_7_bis(i)
        diference = step_8(i)
        if diference < ERelMax:
            loop2 = False
        else:
            #print("Hacemos Tcal = Test e iteramos loop2")
            Test[i+1] = Tcal[i+1]
            #input('Press <CR> to continue...')
    #print("Ejecutando polítropo 2")
    politropo_2(i)
    #print("Ejecutando paso 6")
    step_6_bis(i)
    
    decision = r[i+1]
    if decision  >= 2.05300:
        loop1 = False
    else:
        P[i+1] = Pcal[i+1]
        T[i+1] = Tcal[i+1]
        L[i+1] = Lcal[i+1]
        M[i+1] = Mcal[i+1]
        print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
        # pasamos a la siguiente capa
        i += 1
        Counter.append(i)

print("-" * 78)
print("Hemos llegado a la zona radiativa.")
print("-" * 78)


# ##  Interpolación 

# In[23]:



last = len(Counter) + 2
prev = len(Counter) + 1

r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')


# In[24]:


err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')

#Como la luminosidad integrando desde el centro sale muy bajo, podemos usar una Tc más alta para reducir el error


# ## Temperatura crítica $T_c=1.6$

# In[42]:


RIni = 0.0
h = 0.108 #paso de integración
Tc = 1.7 #Temperatura en el centro

T=np.zeros(num_capas)
Test=np.zeros(num_capas)
Tcal=np.zeros(num_capas)

P=np.zeros(num_capas)
Pest=np.zeros(num_capas)
Pcal=np.zeros(num_capas)

fp=np.zeros(num_capas)
ft=np.zeros(num_capas)
fm=np.zeros(num_capas)
fl=np.zeros(num_capas)


M=np.zeros(num_capas)
Mcal=np.zeros(num_capas)
Mest=np.zeros(num_capas)

L=np.zeros(num_capas)
Lcal=np.zeros(num_capas)
GenEnergiaPP=np.zeros(num_capas)
GenEnergiaCN=np.zeros(num_capas)

r = np.zeros(num_capas) #arrays vacios para almacenar los datos


novotny = np.ones(num_capas) * 2.5
print(h)


# In[43]:


i=-1

while i<=1:
    i += 1
    Tres_primeras_capas_cen(i)
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
    #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')


# In[44]:


#
# Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
#

i = 2  # número de la última capa calculada
Counter = []
print("-" * 78)
print("Algoritmo A.2.")
print("-" * 78)

# Utilizamos las variables lógicas loop1, loop2 y loop3
# para controlar cada uno de los bucle necesarios para
# programar el algoritmo de la fase A.1. (envoltura radiativa)

loop1 = True
while loop1:
    #print("\n*Calculando capa número:", i+1, "\n")
    #print("Ejecutando paso 1")
    r[i+1] = RIni+(i+1) * h
    #print("Ejecutando paso 2 bis")
    step_2_bis(i)
    
    loop2 = True
    while loop2:
        #print("Ejecutando polítropo 1")

        politropo_1(i)
            
        #print("Ejecutando paso 3")
        step_3(i)
            
            
        #print("Ejecutando paso 7 bis")
        
        step_7_bis(i)
        diference = step_8(i)
        if diference < ERelMax:
            loop2 = False
        else:
            #print("Hacemos Tcal = Test e iteramos loop2")
            Test[i+1] = Tcal[i+1]
            #input('Press <CR> to continue...')
    #print("Ejecutando polítropo 2")
    politropo_2(i)
    #print("Ejecutando paso 6")
    step_6_bis(i)
    
    decision = r[i+1]
    if decision >= 2.053:
        loop1 = False
    else:
        P[i+1] = Pcal[i+1]
        T[i+1] = Tcal[i+1]
        L[i+1] = Lcal[i+1]
        M[i+1] = Mcal[i+1]
        print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
        # pasamos a la siguiente capa
        i += 1
        Counter.append(i)

print("-" * 78)
print("Hemos llegado a la zona convectiva.")
print("-" * 78)


# In[45]:


last = len(Counter) + 2
prev = len(Counter) + 1
r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
    #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
    #print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')


# In[46]:


err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')


# ## Temperatura crítica $T_c=1.86$

# In[59]:


RIni = 0.0
h = 0.108 #paso de integración
Tc = 1.85 #Temperatura en el centro

T=np.zeros(num_capas)
Test=np.zeros(num_capas)
Tcal=np.zeros(num_capas)

P=np.zeros(num_capas)
Pest=np.zeros(num_capas)
Pcal=np.zeros(num_capas)

fp=np.zeros(num_capas)
ft=np.zeros(num_capas)
fm=np.zeros(num_capas)
fl=np.zeros(num_capas)


M=np.zeros(num_capas)
Mcal=np.zeros(num_capas)
Mest=np.zeros(num_capas)

L=np.zeros(num_capas)
Lcal=np.zeros(num_capas)
GenEnergiaPP=np.zeros(num_capas)
GenEnergiaCN=np.zeros(num_capas)

r = np.zeros(num_capas) #arrays vacios para almacenar los datos


novotny = np.ones(num_capas) * 2.5
print(h)


# In[60]:



i=-1

while i<=1:
    i += 1
    Tres_primeras_capas_cen(i)
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
    #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')


# In[61]:


#
# Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
#

i = 2  # número de la última capa calculada
Counter = []
print("-" * 78)
print("Algoritmo A.2.")
print("-" * 78)

# Utilizamos las variables lógicas loop1, loop2 y loop3
# para controlar cada uno de los bucle necesarios para
# programar el algoritmo de la fase A.1. (envoltura radiativa)

loop1 = True
while loop1:
    #print("\n*Calculando capa número:", i+1, "\n")
    #print("Ejecutando paso 1")
    r[i+1] = RIni+(i+1) * h
    #print("Ejecutando paso 2 bis")
    step_2_bis(i)
    
    loop2 = True
    while loop2:
        #print("Ejecutando polítropo 1")

        politropo_1(i)
            
        #print("Ejecutando paso 3")
        step_3(i)
            
            
        #print("Ejecutando paso 7 bis")
        
        step_7_bis(i)
        diference = step_8(i)
        if diference < ERelMax:
            loop2 = False
        else:
            #print("Hacemos Tcal = Test e iteramos loop2")
            Test[i+1] = Tcal[i+1]
            #input('Press <CR> to continue...')
    #print("Ejecutando polítropo 2")
    politropo_2(i)
    #print("Ejecutando paso 6")
    step_6_bis(i)
    
    decision = r[i+1]
    if decision >= 2.053:
        loop1 = False
    else:
        P[i+1] = Pcal[i+1]
        T[i+1] = Tcal[i+1]
        L[i+1] = Lcal[i+1]
        M[i+1] = Mcal[i+1]
        print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
        # pasamos a la siguiente capa
        i += 1
        Counter.append(i)

print("-" * 78)
print("Hemos llegado a la zona convectiva.")
print("-" * 78)


# In[62]:


last = len(Counter) + 2
prev = len(Counter) + 1


# In[63]:


r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')


# In[64]:


err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')


# ## Bucle para $T_c$ entre 1.8 y 1.87

# In[36]:


RIni = 0.0
h = 0.108 #paso de integración


T=np.zeros(num_capas)
Test=np.zeros(num_capas)
Tcal=np.zeros(num_capas)

P=np.zeros(num_capas)
Pest=np.zeros(num_capas)
Pcal=np.zeros(num_capas)

fp=np.zeros(num_capas)
ft=np.zeros(num_capas)
fm=np.zeros(num_capas)
fl=np.zeros(num_capas)


M=np.zeros(num_capas)
Mcal=np.zeros(num_capas)
Mest=np.zeros(num_capas)

L=np.zeros(num_capas)
Lcal=np.zeros(num_capas)
GenEnergiaPP=np.zeros(num_capas)
GenEnergiaCN=np.zeros(num_capas)

r = np.zeros(num_capas) #arrays vacios para almacenar los datos


novotny = np.ones(num_capas) * 2.5
print(x)
print(h)


# In[37]:


def Tres_primeras_capas_cen(i):
    r[i]=RIni+(i*h)
    M[i]=M_tres_primeras_capas(Tc, mu,k,r[i])
    L[i]=L_tres_primeras_capas(mu,k,Tc,r[i])
    T[i]=T_conveccion(Tc, mu, k, r[i])
    P[i]=P_tres_primeras_capas(k, T[i])
    fm[i]=fi_m_cen(k,T[i],r[i])
    fl[i]=fi_l_cen(k,T[i],r[i],mu)
    ft[i]=fi_t_cen(M[i],r[i],mu)
    fp[i]=fi_p_cen(k,T[i],M[i],r[i])
    


# In[38]:



Tc = 1.8
temp = []
ERROR = []
while Tc <=1.9:
    Tc += 0.0001
    
     #arrays vacios para almacenar los datos
    
    i=-1
    Counter = []

    while i<=1:
        i += 1
        Tres_primeras_capas_cen(i)
        #print(f'{Tc:8.5f} {i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
        #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')
        
    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #

    i = 2  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.2.")
    #print("-" * 78)

    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+(i+1) * h
        #print("Ejecutando paso 2 bis")
        step_2_bis(i)

        loop2 = True
        while loop2:
            #print("Ejecutando polítropo 1")

            politropo_1(i)

            #print("Ejecutando paso 3")
            step_3(i)


            #print("Ejecutando paso 7 bis")

            step_7_bis(i)
            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
                #input('Press <CR> to continue...')
        #print("Ejecutando polítropo 2")
        politropo_2(i)
        #print("Ejecutando paso 6")
        step_6_bis(i)

        decision = r[i+1]
        if decision >= 2.053:
            loop1 = False
        else:
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]
            #print(f'{Tc:8.5f} {i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
            # pasamos a la siguiente capa
            i += 1

    #print("-" * 78)
    #print("Hemos llegado a la zona convectiva.")
    #print("-" * 78)
    
    r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
    P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
    T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
    L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
    M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
    
    err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
    err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
    err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
    err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
    err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
    ERROR.append(err_Tot)
    temp.append(Tc)
    
    #print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')
        
        


# In[66]:


min_err = min(ERROR)
print(min_err)

index = ERROR.index(min(ERROR))
min_temp = temp[index]
print(min_temp)
sns.set_style("whitegrid")
plt.plot(temp,ERROR)
plt.plot(min_temp,min_err,marker='o',color='red')
plt.annotate("21,65%,1.8572",(min_temp,min_err))
plt.xlabel('Temperatura central($10^7 K$)')
plt.ylabel('Error relativo(%)')
plt.title('Error relativo en funcion de Tc')
plt.savefig('Errores.pdf')
plt.show()


# ## Modelo completo

# In[40]:


RIni = 12
h  = -0.108
T=np.zeros(num_capas)
Test=np.zeros(num_capas)
Tcal=np.zeros(num_capas)

P=np.zeros(num_capas)
Pest=np.zeros(num_capas)
Pcal=np.zeros(num_capas)

fp=np.zeros(num_capas)
ft=np.zeros(num_capas)
fm=np.zeros(num_capas)
fl=np.zeros(num_capas)


M=np.zeros(num_capas)
Mcal=np.zeros(num_capas)
Mest=np.zeros(num_capas)

L=np.zeros(num_capas)
Lcal=np.zeros(num_capas)
GenEnergiaPP=np.zeros(num_capas)
GenEnergiaCN=np.zeros(num_capas)

r = np.zeros(num_capas) #arrays vacios para almacenar los datos
n=np.zeros(num_capas)
novotny = np.ones(num_capas) * 2.5

def Tres_primeras_capas(i):
    r[i]=RIni+(i*h)
    T[i]=T_primeras_capas(A1, r[i], RTot)
    P[i]=P_primeras_capas(A2, T[i])
    L[i]=LTot
    M[i]=MTot
    fp[i]=f_i_presion(Cp, P[i], T[i], MTot, r[i])
    ft[i]=f_i_temperatura(Ct, P[i], LTot, T[i], r[i])
    fm[i]=0.0
    fl[i]=0.0
    
    if DEBUG:
        print(f'i: {i}, fp[i]: {fp[i]}')
        print(f'i: {i}, ft[i]: {ft[i]}')
        print(f'i: {i}, fl[i]: {fl[i]}')
        print(f'i: {i}, fm[i]: {fm[i]}')  


# In[41]:


i=-1
while i<=9:
    i += 1
    Tres_primeras_capas(i)
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')


# In[42]:


MTot = 5  #masa total
LTot = 40 #luminosidd total
RTot = 12 #radio total
RIni = 0.9*RTot #radio inicial
num_capas = 101 #número de capas
X = 0.8 #contenido de hidrogeno
Y = 0.16 #contenido de helio
Z = 1-Y-X #elementos pesados
mu = 1/((2*X)+(3*Y/4)+(Z/2)) #peso molecular medio
A1 = 1.9022*mu*MTot #cte para la temp de las primeras capas
A2 = 10.645*np.sqrt(MTot/(mu*Z*LTot*(1+X))) #cte para la pres de las primeras capas
Cp = 8.084*mu #cte para calcular f_i de las pres
Ct = 0.01679*Z*(1+X)*mu**2 #cte para calcular f_i de la temp
h = -RIni/(num_capas-1) #paso de integración
ERelMax = 0.0001
X1_PP = X
X2_PP = X
X1_CN = X
X2_CN = Z / 3
rho = MTot / (4/3) * np.pi * RTot**3

T=np.zeros(num_capas)
Test=np.zeros(num_capas)
Tcal=np.zeros(num_capas)

P=np.zeros(num_capas)
Pest=np.zeros(num_capas)
Pcal=np.zeros(num_capas)

fp=np.zeros(num_capas)
ft=np.zeros(num_capas)
fm=np.zeros(num_capas)
fl=np.zeros(num_capas)


M=np.zeros(num_capas)
Mcal=np.zeros(num_capas)
Mest=np.zeros(num_capas)

L=np.zeros(num_capas)
Lcal=np.zeros(num_capas)
GenEnergiaPP=np.zeros(num_capas)
GenEnergiaCN=np.zeros(num_capas)

r = np.zeros(num_capas) #arrays vacios para almacenar los datos
n=np.zeros(num_capas)
novotny = np.ones(num_capas) * 2.5


# In[43]:


i=-1

while i<=1:
    i += 1
    Tres_primeras_capas(i)
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
    
# Ya tenemos las tres primeras capas.    


# In[44]:


#
# Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
#

i = 2  # número de la última capa calculada

print("-" * 78)
print("Algoritmo A.1.")
print("-" * 78)

# Utilizamos las variables lógicas loop1, loop2 y loop3
# para controlar cada uno de los bucle necesarios para
# programar el algoritmo de la fase A.1. (envoltura radiativa)

loop1 = True
while loop1:
    #print("\n*Calculando capa número:", i+1, "\n")
    #print("Ejecutando paso 1")
    r[i+1] = RIni+ (i+1)*h
    #print("Ejecutando paso 2")
    step_2(i)    
    loop2 = True
    while loop2:
        loop3 = True
        while loop3:
            #print("Ejecutando paso 3")
            step_3(i)
            #print("Ejecutando paso 4")
            step_4(i)
            
            #print(Pest[i+1], Pcal[i+1])
            diference = step_5(i)
            if diference < ERelMax:
                loop3 = False
            else:
                #print("Hacemos Pest = Pcal e iteramos loop3")
                Pest[i+1] = Pcal[i+1]
                #input('Press <CR> to continue...')
            
        #print("Ejecutando paso 6")
        
        step_6(i)
        
        #print("Ejecutando paso 7")
        step_7(i)
        
        diference = step_8(i)
        if diference < ERelMax:
            loop2 = False
        else:
            #print("Hacemos Tcal = Test e iteramos loop2")
            Test[i+1] = Tcal[i+1]
            #input('Press <CR> to continue...')

    #print("Ejecutando paso 9")
    step_9(i)
    
    decision = n[i+1]  
    if decision <= 2.5:
        loop1 = False
    else:
        # almacenamos los valores calculados y los mostramos
        P[i+1] = Pcal[i+1]
        T[i+1] = Tcal[i+1]
        L[i+1] = Lcal[i+1]
        M[i+1] = Mcal[i+1]
        print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}')   

        # pasamos a la siguiente capa
        i += 1
        
    
    #input('Press <CR> to continue...')


print("-" * 78)
print("Pasamos a la fase convectiva, algoritmo A.2.")
print("-" * 78)


# In[45]:


RIni = 0
h = 0.108 #paso de integración
Tc = temp[index] #Temperatura en el centro

T=np.zeros(num_capas)
Test=np.zeros(num_capas)
Tcal=np.zeros(num_capas)

P=np.zeros(num_capas)
Pest=np.zeros(num_capas)
Pcal=np.zeros(num_capas)

fp=np.zeros(num_capas)
ft=np.zeros(num_capas)
fm=np.zeros(num_capas)
fl=np.zeros(num_capas)


M=np.zeros(num_capas)
Mcal=np.zeros(num_capas)
Mest=np.zeros(num_capas)

L=np.zeros(num_capas)
Lcal=np.zeros(num_capas)
GenEnergiaPP=np.zeros(num_capas)
GenEnergiaCN=np.zeros(num_capas)

r = np.zeros(num_capas) #arrays vacios para almacenar los datos


novotny = np.ones(num_capas) * 2.5


# In[46]:


i=-1

while i<=1:
    i += 1
    Tres_primeras_capas_cen(i)
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
    #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')


# In[48]:


#
# Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
#

i = 2  # número de la última capa calculada

print("-" * 78)
print("Algoritmo A.2.")
print("-" * 78)

# Utilizamos las variables lógicas loop1, loop2 y loop3
# para controlar cada uno de los bucle necesarios para
# programar el algoritmo de la fase A.1. (envoltura radiativa)

loop1 = True
while loop1:
    #print("\n*Calculando capa número:", i+1, "\n")
    #print("Ejecutando paso 1")
    r[i+1] = RIni+(i+1) * h
    #print("Ejecutando paso 2 bis")
    step_2_bis(i)
    
    loop2 = True
    while loop2:
        #print("Ejecutando polítropo 1")

        politropo_1(i)
            
        #print("Ejecutando paso 3")
        step_3(i)
            
            
        #print("Ejecutando paso 7 bis")
        
        step_7_bis(i)
        diference = step_8(i)
        if diference < ERelMax:
            loop2 = False
        else:
            #print("Hacemos Tcal = Test e iteramos loop2")
            Test[i+1] = Tcal[i+1]
            #input('Press <CR> to continue...')
    #print("Ejecutando polítropo 2")
    politropo_2(i)
    #print("Ejecutando paso 6")
    step_6_bis(i)
    
    
    decision = r[i+1]
    if decision >= 2.053:
        loop1 = False
    else:
        P[i+1] = Pcal[i+1]
        T[i+1] = Tcal[i+1]
        L[i+1] = Lcal[i+1]
        M[i+1] = Mcal[i+1]
        print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} ') 
        # pasamos a la siguiente capa
        i += 1

print("-" * 78)
print("Hemos llegado a la zona convectiva.")
print("-" * 78)


#  ## Jugando con $R_{Tot}$ y $L_{Tot}$

# ##  $L_{Tot}=30$

# In[49]:


errores = []


# In[65]:


LTot = 28.5
while LTot <= 44:
    
    LTot += 1.5
    
    RTot = 11
    MTot = 5  #masa total
    RIni = 0.9*RTot #radio inicial
    num_capas = 101 #número de capas
    X = 0.8 #contenido de hidrogeno
    Y = 0.16 #contenido de helio
    Z = 1-Y-X #elementos pesados
    mu = 1/((2*X)+(3*Y/4)+(Z/2)) #peso molecular medio
    A1 = 1.9022*mu*MTot #cte para la temp de las primeras capas
    A2 = 10.645*np.sqrt(MTot/(mu*Z*LTot*(1+X))) #cte para la pres de las primeras capas
    Cp = 8.084*mu #cte para calcular f_i de las pres
    Ct = 0.01679*Z*(1+X)*mu**2 #cte para calcular f_i de la temp
    h = -RIni/(num_capas-1) #paso de integración
    ERelMax = 0.0001
    X1_PP = X
    X2_PP = X
    X1_CN = X
    X2_CN = Z / 3
    rho = MTot / (4/3) * np.pi * RTot**3

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos
    n=np.zeros(num_capas)

    novotny = np.ones(num_capas) * 2.5

    i=-1

    while i<=1:
        i += 1
        Tres_primeras_capas(i)
        #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')



    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #

    i = 2  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.1.")
    #print("-" * 78)
    Counter = []
    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+ (i+1)*h
        #print("Ejecutando paso 2")
        step_2(i)    
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                #print("Ejecutando paso 3")
                step_3(i)
                #print("Ejecutando paso 4")
                step_4(i)

                #print(Pest[i+1], Pcal[i+1])
                diference = step_5(i)
                if diference < ERelMax:
                    loop3 = False
                else:
                    #print("Hacemos Pest = Pcal e iteramos loop3")
                    Pest[i+1] = Pcal[i+1]
                    #input('Press <CR> to continue...')

            #print("Ejecutando paso 6")

            step_6(i)

            #print("Ejecutando paso 7")
            step_7(i)

            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
                #input('Press <CR> to continue...')

        #print("Ejecutando paso 9")
        step_9(i)

        decision = n[i+1]  
        if decision <= 2.5:
            loop1 = False
        else:
            # almacenamos los valores calculados y los mostramos
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]
            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}')   

            # pasamos a la siguiente capa
            i += 1
            Counter.append(i)



        #input('Press <CR> to continue...')


    #print("-" * 78)
    #print("Pasamos a la fase convectiva, algoritmo A.2.")
    #print("-" * 78)


    # Ya tenemos las tres primeras capas.   

    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #
    last_layer = len(Counter) + 2
    next_layer = len(Counter) + 3
    x = r[last_layer]
    #print(x)
    i = last_layer  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.2.")
    #print("-" * 78)

    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+(i+1) * h
        step_9_bis(i)
        #print("Ejecutando paso 2 bis")
        step_2_bis(i)

        loop2 = True
        while loop2:
            #print("Ejecutando polítropo 1")

            politropo_1(i)

            #print("Ejecutando paso 3")
            step_3(i)


            #print("Ejecutando paso 7 bis")

            step_7_bis(i)
            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
        #print("Ejecutando polítropo 2")
        politropo_2(i)

        #print("Ejecutando paso 6")
        step_6_bis(i)


        decision = r[i+1]

        if decision <= 0:
            loop1 = False
        else:
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]

            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {n[i+1]:10.6f}') 
            # pasamos a la siguiente capa
            i += 1

    #print("-" * 78)
    #print("Hemos llegado al nucleo de la estrella.")
    #print("-" * 78)

    r_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last_layer] - r[next_layer])) + r[next_layer]
    P_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last_layer] - P[next_layer])) + P[next_layer]
    T_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last_layer] - T[next_layer])) + T[next_layer]
    L_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last_layer] - L[next_layer])) + L[next_layer]
    M_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last_layer] - M[next_layer])) + M[next_layer]
    #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')


    RIni = 0.0
    h = -h #paso de integración

    

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos

    Tc = 1.8
    temp = []
    ERROR = []
    while Tc <=1.9:
        Tc += 0.0001
        i=-1

        while i<=1:
            i += 1
            Tres_primeras_capas_cen(i)
            #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
            #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')
        #
        # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
        #

        i = 2  # número de la última capa calculada
        Counter = []
        #print("-" * 78)
        #print("Algoritmo A.2.")
        #print("-" * 78)

        # Utilizamos las variables lógicas loop1, loop2 y loop3
        # para controlar cada uno de los bucle necesarios para
        # programar el algoritmo de la fase A.1. (envoltura radiativa)

        loop1 = True
        while loop1:
            #print("\n*Calculando capa número:", i+1, "\n")
            #print("Ejecutando paso 1")
            r[i+1] = RIni+(i+1) * h
            #print("Ejecutando paso 2 bis")
            step_2_bis(i)

            loop2 = True
            while loop2:
                #print("Ejecutando polítropo 1")

                politropo_1(i)

                #print("Ejecutando paso 3")
                step_3(i)


                #print("Ejecutando paso 7 bis")

                step_7_bis(i)
                diference = step_8(i)
                if diference < ERelMax:
                    loop2 = False
                else:
                    #print("Hacemos Tcal = Test e iteramos loop2")
                    Test[i+1] = Tcal[i+1]
                    #input('Press <CR> to continue...')
            #print("Ejecutando polítropo 2")
            politropo_2(i)
            #print("Ejecutando paso 6")
            step_6_bis(i)

            decision = r[i+1]
            if decision >= 2.053:
                loop1 = False
            else:
                P[i+1] = Pcal[i+1]
                T[i+1] = Tcal[i+1]
                L[i+1] = Lcal[i+1]
                M[i+1] = Mcal[i+1]
                #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
                # pasamos a la siguiente capa
                i += 1
                Counter.append(i)


        #print("-" * 78)
        #print("Hemos llegado a la zona convectiva.")
        #print("-" * 78)

        last = len(Counter) + 2
        prev = len(Counter) + 1
        r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
    P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
    T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
    L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
    M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
            #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
        #print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')

    err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
    err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
    err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
    err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
    err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
    ERROR.append(err_Tot)
    temp.append(Tc)
    
        #print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')
    print(f'{LTot:8.1f} {min(ERROR):8.2f} ')
    
   
    


# In[66]:


LTot = 28.5
while LTot <= 44:
    
    LTot += 1.5
    
    RTot = 11.1
    MTot = 5  #masa total
    RIni = 0.9*RTot #radio inicial
    num_capas = 101 #número de capas
    X = 0.8 #contenido de hidrogeno
    Y = 0.16 #contenido de helio
    Z = 1-Y-X #elementos pesados
    mu = 1/((2*X)+(3*Y/4)+(Z/2)) #peso molecular medio
    A1 = 1.9022*mu*MTot #cte para la temp de las primeras capas
    A2 = 10.645*np.sqrt(MTot/(mu*Z*LTot*(1+X))) #cte para la pres de las primeras capas
    Cp = 8.084*mu #cte para calcular f_i de las pres
    Ct = 0.01679*Z*(1+X)*mu**2 #cte para calcular f_i de la temp
    h = -RIni/(num_capas-1) #paso de integración
    ERelMax = 0.0001
    X1_PP = X
    X2_PP = X
    X1_CN = X
    X2_CN = Z / 3
    rho = MTot / (4/3) * np.pi * RTot**3

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos
    n=np.zeros(num_capas)

    novotny = np.ones(num_capas) * 2.5

    i=-1

    while i<=1:
        i += 1
        Tres_primeras_capas(i)
        #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')



    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #

    i = 2  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.1.")
    #print("-" * 78)
    Counter = []
    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+ (i+1)*h
        #print("Ejecutando paso 2")
        step_2(i)    
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                #print("Ejecutando paso 3")
                step_3(i)
                #print("Ejecutando paso 4")
                step_4(i)

                #print(Pest[i+1], Pcal[i+1])
                diference = step_5(i)
                if diference < ERelMax:
                    loop3 = False
                else:
                    #print("Hacemos Pest = Pcal e iteramos loop3")
                    Pest[i+1] = Pcal[i+1]
                    #input('Press <CR> to continue...')

            #print("Ejecutando paso 6")

            step_6(i)

            #print("Ejecutando paso 7")
            step_7(i)

            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
                #input('Press <CR> to continue...')

        #print("Ejecutando paso 9")
        step_9(i)

        decision = n[i+1]  
        if decision <= 2.5:
            loop1 = False
        else:
            # almacenamos los valores calculados y los mostramos
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]
            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}')   

            # pasamos a la siguiente capa
            i += 1
            Counter.append(i)



        #input('Press <CR> to continue...')


    #print("-" * 78)
    #print("Pasamos a la fase convectiva, algoritmo A.2.")
    #print("-" * 78)


    # Ya tenemos las tres primeras capas.   

    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #
    last_layer = len(Counter) + 2
    next_layer = len(Counter) + 3
    x = r[last_layer]
    #print(x)
    i = last_layer  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.2.")
    #print("-" * 78)

    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+(i+1) * h
        step_9_bis(i)
        #print("Ejecutando paso 2 bis")
        step_2_bis(i)

        loop2 = True
        while loop2:
            #print("Ejecutando polítropo 1")

            politropo_1(i)

            #print("Ejecutando paso 3")
            step_3(i)


            #print("Ejecutando paso 7 bis")

            step_7_bis(i)
            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
        #print("Ejecutando polítropo 2")
        politropo_2(i)

        #print("Ejecutando paso 6")
        step_6_bis(i)


        decision = r[i+1]

        if decision <= 0:
            loop1 = False
        else:
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]

            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {n[i+1]:10.6f}') 
            # pasamos a la siguiente capa
            i += 1

    #print("-" * 78)
    #print("Hemos llegado al nucleo de la estrella.")
    #print("-" * 78)

    r_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last_layer] - r[next_layer])) + r[next_layer]
    P_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last_layer] - P[next_layer])) + P[next_layer]
    T_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last_layer] - T[next_layer])) + T[next_layer]
    L_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last_layer] - L[next_layer])) + L[next_layer]
    M_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last_layer] - M[next_layer])) + M[next_layer]
    #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')


    RIni = 0.0
    h = -h #paso de integración

    

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos

    Tc = 1.8
    temp = []
    ERROR = []
    while Tc <=1.9:
        Tc += 0.0001
        i=-1

        while i<=1:
            i += 1
            Tres_primeras_capas_cen(i)
            #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
            #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')
        #
        # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
        #

        i = 2  # número de la última capa calculada
        Counter = []
        #print("-" * 78)
        #print("Algoritmo A.2.")
        #print("-" * 78)

        # Utilizamos las variables lógicas loop1, loop2 y loop3
        # para controlar cada uno de los bucle necesarios para
        # programar el algoritmo de la fase A.1. (envoltura radiativa)

        loop1 = True
        while loop1:
            #print("\n*Calculando capa número:", i+1, "\n")
            #print("Ejecutando paso 1")
            r[i+1] = RIni+(i+1) * h
            #print("Ejecutando paso 2 bis")
            step_2_bis(i)

            loop2 = True
            while loop2:
                #print("Ejecutando polítropo 1")

                politropo_1(i)

                #print("Ejecutando paso 3")
                step_3(i)


                #print("Ejecutando paso 7 bis")

                step_7_bis(i)
                diference = step_8(i)
                if diference < ERelMax:
                    loop2 = False
                else:
                    #print("Hacemos Tcal = Test e iteramos loop2")
                    Test[i+1] = Tcal[i+1]
                    #input('Press <CR> to continue...')
            #print("Ejecutando polítropo 2")
            politropo_2(i)
            #print("Ejecutando paso 6")
            step_6_bis(i)

            decision = r[i+1]
            if decision >= 2.053:
                loop1 = False
            else:
                P[i+1] = Pcal[i+1]
                T[i+1] = Tcal[i+1]
                L[i+1] = Lcal[i+1]
                M[i+1] = Mcal[i+1]
                #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
                # pasamos a la siguiente capa
                i += 1
                Counter.append(i)


        #print("-" * 78)
        #print("Hemos llegado a la zona convectiva.")
        #print("-" * 78)

        last = len(Counter) + 2
        prev = len(Counter) + 1
        r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
        P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
        T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
        L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
        M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
        #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
        #print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')

        err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
        err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
        err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
        err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
        err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
        ERROR.append(err_Tot)
        temp.append(Tc)
        #print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')
    print(f'{LTot:8.1f} {min(ERROR):8.2f} ')
    


# In[67]:


LTot = 28.5
while LTot <= 44:
    
    LTot += 1.5
    
    RTot = 11.2
    MTot = 5  #masa total
    RIni = 0.9*RTot #radio inicial
    num_capas = 101 #número de capas
    X = 0.8 #contenido de hidrogeno
    Y = 0.16 #contenido de helio
    Z = 1-Y-X #elementos pesados
    mu = 1/((2*X)+(3*Y/4)+(Z/2)) #peso molecular medio
    A1 = 1.9022*mu*MTot #cte para la temp de las primeras capas
    A2 = 10.645*np.sqrt(MTot/(mu*Z*LTot*(1+X))) #cte para la pres de las primeras capas
    Cp = 8.084*mu #cte para calcular f_i de las pres
    Ct = 0.01679*Z*(1+X)*mu**2 #cte para calcular f_i de la temp
    h = -RIni/(num_capas-1) #paso de integración
    ERelMax = 0.0001
    X1_PP = X
    X2_PP = X
    X1_CN = X
    X2_CN = Z / 3
    rho = MTot / (4/3) * np.pi * RTot**3

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos
    n=np.zeros(num_capas)

    novotny = np.ones(num_capas) * 2.5

    i=-1

    while i<=1:
        i += 1
        Tres_primeras_capas(i)
        #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')



    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #

    i = 2  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.1.")
    #print("-" * 78)
    Counter = []
    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+ (i+1)*h
        #print("Ejecutando paso 2")
        step_2(i)    
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                #print("Ejecutando paso 3")
                step_3(i)
                #print("Ejecutando paso 4")
                step_4(i)

                #print(Pest[i+1], Pcal[i+1])
                diference = step_5(i)
                if diference < ERelMax:
                    loop3 = False
                else:
                    #print("Hacemos Pest = Pcal e iteramos loop3")
                    Pest[i+1] = Pcal[i+1]
                    #input('Press <CR> to continue...')

            #print("Ejecutando paso 6")

            step_6(i)

            #print("Ejecutando paso 7")
            step_7(i)

            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
                #input('Press <CR> to continue...')

        #print("Ejecutando paso 9")
        step_9(i)

        decision = n[i+1]  
        if decision <= 2.5:
            loop1 = False
        else:
            # almacenamos los valores calculados y los mostramos
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]
            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}')   

            # pasamos a la siguiente capa
            i += 1
            Counter.append(i)



        #input('Press <CR> to continue...')


    #print("-" * 78)
    #print("Pasamos a la fase convectiva, algoritmo A.2.")
    #print("-" * 78)


    # Ya tenemos las tres primeras capas.   

    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #
    last_layer = len(Counter) + 2
    next_layer = len(Counter) + 3
    x = r[last_layer]
    #print(x)
    i = last_layer  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.2.")
    #print("-" * 78)

    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+(i+1) * h
        step_9_bis(i)
        #print("Ejecutando paso 2 bis")
        step_2_bis(i)

        loop2 = True
        while loop2:
            #print("Ejecutando polítropo 1")

            politropo_1(i)

            #print("Ejecutando paso 3")
            step_3(i)


            #print("Ejecutando paso 7 bis")

            step_7_bis(i)
            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
        #print("Ejecutando polítropo 2")
        politropo_2(i)

        #print("Ejecutando paso 6")
        step_6_bis(i)


        decision = r[i+1]

        if decision <= 0:
            loop1 = False
        else:
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]

            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {n[i+1]:10.6f}') 
            # pasamos a la siguiente capa
            i += 1

    #print("-" * 78)
    #print("Hemos llegado al nucleo de la estrella.")
    #print("-" * 78)

    r_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last_layer] - r[next_layer])) + r[next_layer]
    P_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last_layer] - P[next_layer])) + P[next_layer]
    T_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last_layer] - T[next_layer])) + T[next_layer]
    L_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last_layer] - L[next_layer])) + L[next_layer]
    M_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last_layer] - M[next_layer])) + M[next_layer]
    #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')


    RIni = 0.0
    h = -h #paso de integración

    

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos

    Tc = 1.8
    temp = []
    ERROR = []
    while Tc <=1.9:
        Tc += 0.0001
        i=-1

        while i<=1:
            i += 1
            Tres_primeras_capas_cen(i)
            #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
            #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')
        #
        # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
        #

        i = 2  # número de la última capa calculada
        Counter = []
        #print("-" * 78)
        #print("Algoritmo A.2.")
        #print("-" * 78)

        # Utilizamos las variables lógicas loop1, loop2 y loop3
        # para controlar cada uno de los bucle necesarios para
        # programar el algoritmo de la fase A.1. (envoltura radiativa)

        loop1 = True
        while loop1:
            #print("\n*Calculando capa número:", i+1, "\n")
            #print("Ejecutando paso 1")
            r[i+1] = RIni+(i+1) * h
            #print("Ejecutando paso 2 bis")
            step_2_bis(i)

            loop2 = True
            while loop2:
                #print("Ejecutando polítropo 1")

                politropo_1(i)

                #print("Ejecutando paso 3")
                step_3(i)


                #print("Ejecutando paso 7 bis")

                step_7_bis(i)
                diference = step_8(i)
                if diference < ERelMax:
                    loop2 = False
                else:
                    #print("Hacemos Tcal = Test e iteramos loop2")
                    Test[i+1] = Tcal[i+1]
                    #input('Press <CR> to continue...')
            #print("Ejecutando polítropo 2")
            politropo_2(i)
            #print("Ejecutando paso 6")
            step_6_bis(i)

            decision = r[i+1]
            if decision >= 2.053:
                loop1 = False
            else:
                P[i+1] = Pcal[i+1]
                T[i+1] = Tcal[i+1]
                L[i+1] = Lcal[i+1]
                M[i+1] = Mcal[i+1]
                #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
                # pasamos a la siguiente capa
                i += 1
                Counter.append(i)


        #print("-" * 78)
        #print("Hemos llegado a la zona convectiva.")
        #print("-" * 78)

        last = len(Counter) + 2
        prev = len(Counter) + 1
        r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
        P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
        T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
        L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
        M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
        #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
        #print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')

        err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
        err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
        err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
        err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
        err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
        ERROR.append(err_Tot)
        temp.append(Tc)
        #print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')
    print(f'{LTot:8.1f} {min(ERROR):8.2f} ')
    


# In[68]:


LTot = 28.5
while LTot <= 44:
    
    LTot += 1.5
    
    RTot = 11.3
    MTot = 5  #masa total
    RIni = 0.9*RTot #radio inicial
    num_capas = 101 #número de capas
    X = 0.8 #contenido de hidrogeno
    Y = 0.16 #contenido de helio
    Z = 1-Y-X #elementos pesados
    mu = 1/((2*X)+(3*Y/4)+(Z/2)) #peso molecular medio
    A1 = 1.9022*mu*MTot #cte para la temp de las primeras capas
    A2 = 10.645*np.sqrt(MTot/(mu*Z*LTot*(1+X))) #cte para la pres de las primeras capas
    Cp = 8.084*mu #cte para calcular f_i de las pres
    Ct = 0.01679*Z*(1+X)*mu**2 #cte para calcular f_i de la temp
    h = -RIni/(num_capas-1) #paso de integración
    ERelMax = 0.0001
    X1_PP = X
    X2_PP = X
    X1_CN = X
    X2_CN = Z / 3
    rho = MTot / (4/3) * np.pi * RTot**3

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos
    n=np.zeros(num_capas)

    novotny = np.ones(num_capas) * 2.5

    i=-1

    while i<=1:
        i += 1
        Tres_primeras_capas(i)
        #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')



    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #

    i = 2  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.1.")
    #print("-" * 78)
    Counter = []
    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+ (i+1)*h
        #print("Ejecutando paso 2")
        step_2(i)    
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                #print("Ejecutando paso 3")
                step_3(i)
                #print("Ejecutando paso 4")
                step_4(i)

                #print(Pest[i+1], Pcal[i+1])
                diference = step_5(i)
                if diference < ERelMax:
                    loop3 = False
                else:
                    #print("Hacemos Pest = Pcal e iteramos loop3")
                    Pest[i+1] = Pcal[i+1]
                    #input('Press <CR> to continue...')

            #print("Ejecutando paso 6")

            step_6(i)

            #print("Ejecutando paso 7")
            step_7(i)

            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
                #input('Press <CR> to continue...')

        #print("Ejecutando paso 9")
        step_9(i)

        decision = n[i+1]  
        if decision <= 2.5:
            loop1 = False
        else:
            # almacenamos los valores calculados y los mostramos
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]
            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}')   

            # pasamos a la siguiente capa
            i += 1
            Counter.append(i)



        #input('Press <CR> to continue...')


    #print("-" * 78)
    #print("Pasamos a la fase convectiva, algoritmo A.2.")
    #print("-" * 78)


    # Ya tenemos las tres primeras capas.   

    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #
    last_layer = len(Counter) + 2
    next_layer = len(Counter) + 3
    x = r[last_layer]
    #print(x)
    i = last_layer  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.2.")
    #print("-" * 78)

    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+(i+1) * h
        step_9_bis(i)
        #print("Ejecutando paso 2 bis")
        step_2_bis(i)

        loop2 = True
        while loop2:
            #print("Ejecutando polítropo 1")

            politropo_1(i)

            #print("Ejecutando paso 3")
            step_3(i)


            #print("Ejecutando paso 7 bis")

            step_7_bis(i)
            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
        #print("Ejecutando polítropo 2")
        politropo_2(i)

        #print("Ejecutando paso 6")
        step_6_bis(i)


        decision = r[i+1]

        if decision <= 0:
            loop1 = False
        else:
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]

            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {n[i+1]:10.6f}') 
            # pasamos a la siguiente capa
            i += 1

    #print("-" * 78)
    #print("Hemos llegado al nucleo de la estrella.")
    #print("-" * 78)

    r_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last_layer] - r[next_layer])) + r[next_layer]
    P_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last_layer] - P[next_layer])) + P[next_layer]
    T_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last_layer] - T[next_layer])) + T[next_layer]
    L_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last_layer] - L[next_layer])) + L[next_layer]
    M_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last_layer] - M[next_layer])) + M[next_layer]
    #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')


    RIni = 0.0
    h = -h #paso de integración

    

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos

    Tc = 1.8
    temp = []
    ERROR = []
    while Tc <=1.9:
        Tc += 0.0001
        i=-1

        while i<=1:
            i += 1
            Tres_primeras_capas_cen(i)
            #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
            #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')
        #
        # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
        #

        i = 2  # número de la última capa calculada
        Counter = []
        #print("-" * 78)
        #print("Algoritmo A.2.")
        #print("-" * 78)

        # Utilizamos las variables lógicas loop1, loop2 y loop3
        # para controlar cada uno de los bucle necesarios para
        # programar el algoritmo de la fase A.1. (envoltura radiativa)

        loop1 = True
        while loop1:
            #print("\n*Calculando capa número:", i+1, "\n")
            #print("Ejecutando paso 1")
            r[i+1] = RIni+(i+1) * h
            #print("Ejecutando paso 2 bis")
            step_2_bis(i)

            loop2 = True
            while loop2:
                #print("Ejecutando polítropo 1")

                politropo_1(i)

                #print("Ejecutando paso 3")
                step_3(i)


                #print("Ejecutando paso 7 bis")

                step_7_bis(i)
                diference = step_8(i)
                if diference < ERelMax:
                    loop2 = False
                else:
                    #print("Hacemos Tcal = Test e iteramos loop2")
                    Test[i+1] = Tcal[i+1]
                    #input('Press <CR> to continue...')
            #print("Ejecutando polítropo 2")
            politropo_2(i)
            #print("Ejecutando paso 6")
            step_6_bis(i)

            decision = r[i+1]
            if decision >= 2.053:
                loop1 = False
            else:
                P[i+1] = Pcal[i+1]
                T[i+1] = Tcal[i+1]
                L[i+1] = Lcal[i+1]
                M[i+1] = Mcal[i+1]
                #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
                # pasamos a la siguiente capa
                i += 1
                Counter.append(i)


        #print("-" * 78)
        #print("Hemos llegado a la zona convectiva.")
        #print("-" * 78)

        last = len(Counter) + 2
        prev = len(Counter) + 1
        r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
        P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
        T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
        L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
        M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
        #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
        #print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')

        err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
        err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
        err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
        err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
        err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
        ERROR.append(err_Tot)
        temp.append(Tc)
        #print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')
    print(f'{LTot:8.1f} {min(ERROR):8.2f} ')
    


# In[69]:


LTot = 28.5
while LTot <= 44:
    
    LTot += 1.5
    
    RTot = 11.4
    MTot = 5  #masa total
    RIni = 0.9*RTot #radio inicial
    num_capas = 101 #número de capas
    X = 0.8 #contenido de hidrogeno
    Y = 0.16 #contenido de helio
    Z = 1-Y-X #elementos pesados
    mu = 1/((2*X)+(3*Y/4)+(Z/2)) #peso molecular medio
    A1 = 1.9022*mu*MTot #cte para la temp de las primeras capas
    A2 = 10.645*np.sqrt(MTot/(mu*Z*LTot*(1+X))) #cte para la pres de las primeras capas
    Cp = 8.084*mu #cte para calcular f_i de las pres
    Ct = 0.01679*Z*(1+X)*mu**2 #cte para calcular f_i de la temp
    h = -RIni/(num_capas-1) #paso de integración
    ERelMax = 0.0001
    X1_PP = X
    X2_PP = X
    X1_CN = X
    X2_CN = Z / 3
    rho = MTot / (4/3) * np.pi * RTot**3

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos
    n=np.zeros(num_capas)

    novotny = np.ones(num_capas) * 2.5

    i=-1

    while i<=1:
        i += 1
        Tres_primeras_capas(i)
        #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')



    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #

    i = 2  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.1.")
    #print("-" * 78)
    Counter = []
    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+ (i+1)*h
        #print("Ejecutando paso 2")
        step_2(i)    
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                #print("Ejecutando paso 3")
                step_3(i)
                #print("Ejecutando paso 4")
                step_4(i)

                #print(Pest[i+1], Pcal[i+1])
                diference = step_5(i)
                if diference < ERelMax:
                    loop3 = False
                else:
                    #print("Hacemos Pest = Pcal e iteramos loop3")
                    Pest[i+1] = Pcal[i+1]
                    #input('Press <CR> to continue...')

            #print("Ejecutando paso 6")

            step_6(i)

            #print("Ejecutando paso 7")
            step_7(i)

            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
                #input('Press <CR> to continue...')

        #print("Ejecutando paso 9")
        step_9(i)

        decision = n[i+1]  
        if decision <= 2.5:
            loop1 = False
        else:
            # almacenamos los valores calculados y los mostramos
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]
            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}')   

            # pasamos a la siguiente capa
            i += 1
            Counter.append(i)



        #input('Press <CR> to continue...')


    #print("-" * 78)
    #print("Pasamos a la fase convectiva, algoritmo A.2.")
    #print("-" * 78)


    # Ya tenemos las tres primeras capas.   

    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #
    last_layer = len(Counter) + 2
    next_layer = len(Counter) + 3
    x = r[last_layer]
    #print(x)
    i = last_layer  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.2.")
    #print("-" * 78)

    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+(i+1) * h
        step_9_bis(i)
        #print("Ejecutando paso 2 bis")
        step_2_bis(i)

        loop2 = True
        while loop2:
            #print("Ejecutando polítropo 1")

            politropo_1(i)

            #print("Ejecutando paso 3")
            step_3(i)


            #print("Ejecutando paso 7 bis")

            step_7_bis(i)
            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
        #print("Ejecutando polítropo 2")
        politropo_2(i)

        #print("Ejecutando paso 6")
        step_6_bis(i)


        decision = r[i+1]

        if decision <= 0:
            loop1 = False
        else:
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]

            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {n[i+1]:10.6f}') 
            # pasamos a la siguiente capa
            i += 1

    #print("-" * 78)
    #print("Hemos llegado al nucleo de la estrella.")
    #print("-" * 78)

    r_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last_layer] - r[next_layer])) + r[next_layer]
    P_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last_layer] - P[next_layer])) + P[next_layer]
    T_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last_layer] - T[next_layer])) + T[next_layer]
    L_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last_layer] - L[next_layer])) + L[next_layer]
    M_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last_layer] - M[next_layer])) + M[next_layer]
    #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')


    RIni = 0.0
    h = -h #paso de integración

    

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos

    Tc = 1.8
    temp = []
    ERROR = []
    while Tc <=1.9:
        Tc += 0.0001
        i=-1

        while i<=1:
            i += 1
            Tres_primeras_capas_cen(i)
            #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
            #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')
        #
        # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
        #

        i = 2  # número de la última capa calculada
        Counter = []
        #print("-" * 78)
        #print("Algoritmo A.2.")
        #print("-" * 78)

        # Utilizamos las variables lógicas loop1, loop2 y loop3
        # para controlar cada uno de los bucle necesarios para
        # programar el algoritmo de la fase A.1. (envoltura radiativa)

        loop1 = True
        while loop1:
            #print("\n*Calculando capa número:", i+1, "\n")
            #print("Ejecutando paso 1")
            r[i+1] = RIni+(i+1) * h
            #print("Ejecutando paso 2 bis")
            step_2_bis(i)

            loop2 = True
            while loop2:
                #print("Ejecutando polítropo 1")

                politropo_1(i)

                #print("Ejecutando paso 3")
                step_3(i)


                #print("Ejecutando paso 7 bis")

                step_7_bis(i)
                diference = step_8(i)
                if diference < ERelMax:
                    loop2 = False
                else:
                    #print("Hacemos Tcal = Test e iteramos loop2")
                    Test[i+1] = Tcal[i+1]
                    #input('Press <CR> to continue...')
            #print("Ejecutando polítropo 2")
            politropo_2(i)
            #print("Ejecutando paso 6")
            step_6_bis(i)

            decision = r[i+1]
            if decision >= 2.053:
                loop1 = False
            else:
                P[i+1] = Pcal[i+1]
                T[i+1] = Tcal[i+1]
                L[i+1] = Lcal[i+1]
                M[i+1] = Mcal[i+1]
                #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
                # pasamos a la siguiente capa
                i += 1
                Counter.append(i)


        #print("-" * 78)
        #print("Hemos llegado a la zona convectiva.")
        #print("-" * 78)

        last = len(Counter) + 2
        prev = len(Counter) + 1
        r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
        P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
        T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
        L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
        M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
        #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
        #print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')

        err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
        err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
        err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
        err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
        err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
        ERROR.append(err_Tot)
        temp.append(Tc)
        #print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')
    print(f'{LTot:8.1f} {min(ERROR):8.2f} ')

    


# In[70]:


LTot = 28.5
while LTot <= 44:
    
    LTot += 1.5
    
    RTot = 11.5
    MTot = 5  #masa total
    RIni = 0.9*RTot #radio inicial
    num_capas = 101 #número de capas
    X = 0.8 #contenido de hidrogeno
    Y = 0.16 #contenido de helio
    Z = 1-Y-X #elementos pesados
    mu = 1/((2*X)+(3*Y/4)+(Z/2)) #peso molecular medio
    A1 = 1.9022*mu*MTot #cte para la temp de las primeras capas
    A2 = 10.645*np.sqrt(MTot/(mu*Z*LTot*(1+X))) #cte para la pres de las primeras capas
    Cp = 8.084*mu #cte para calcular f_i de las pres
    Ct = 0.01679*Z*(1+X)*mu**2 #cte para calcular f_i de la temp
    h = -RIni/(num_capas-1) #paso de integración
    ERelMax = 0.0001
    X1_PP = X
    X2_PP = X
    X1_CN = X
    X2_CN = Z / 3
    rho = MTot / (4/3) * np.pi * RTot**3

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos
    n=np.zeros(num_capas)

    novotny = np.ones(num_capas) * 2.5

    i=-1

    while i<=1:
        i += 1
        Tres_primeras_capas(i)
        #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')



    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #

    i = 2  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.1.")
    #print("-" * 78)
    Counter = []
    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+ (i+1)*h
        #print("Ejecutando paso 2")
        step_2(i)    
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                #print("Ejecutando paso 3")
                step_3(i)
                #print("Ejecutando paso 4")
                step_4(i)

                #print(Pest[i+1], Pcal[i+1])
                diference = step_5(i)
                if diference < ERelMax:
                    loop3 = False
                else:
                    #print("Hacemos Pest = Pcal e iteramos loop3")
                    Pest[i+1] = Pcal[i+1]
                    #input('Press <CR> to continue...')

            #print("Ejecutando paso 6")

            step_6(i)

            #print("Ejecutando paso 7")
            step_7(i)

            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
                #input('Press <CR> to continue...')

        #print("Ejecutando paso 9")
        step_9(i)

        decision = n[i+1]  
        if decision <= 2.5:
            loop1 = False
        else:
            # almacenamos los valores calculados y los mostramos
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]
            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}')   

            # pasamos a la siguiente capa
            i += 1
            Counter.append(i)



        #input('Press <CR> to continue...')


    #print("-" * 78)
    #print("Pasamos a la fase convectiva, algoritmo A.2.")
    #print("-" * 78)


    # Ya tenemos las tres primeras capas.   

    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #
    last_layer = len(Counter) + 2
    next_layer = len(Counter) + 3
    x = r[last_layer]
    #print(x)
    i = last_layer  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.2.")
    #print("-" * 78)

    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+(i+1) * h
        step_9_bis(i)
        #print("Ejecutando paso 2 bis")
        step_2_bis(i)

        loop2 = True
        while loop2:
            #print("Ejecutando polítropo 1")

            politropo_1(i)

            #print("Ejecutando paso 3")
            step_3(i)


            #print("Ejecutando paso 7 bis")

            step_7_bis(i)
            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
        #print("Ejecutando polítropo 2")
        politropo_2(i)

        #print("Ejecutando paso 6")
        step_6_bis(i)


        decision = r[i+1]

        if decision <= 0:
            loop1 = False
        else:
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]

            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {n[i+1]:10.6f}') 
            # pasamos a la siguiente capa
            i += 1

    #print("-" * 78)
    #print("Hemos llegado al nucleo de la estrella.")
    #print("-" * 78)

    r_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last_layer] - r[next_layer])) + r[next_layer]
    P_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last_layer] - P[next_layer])) + P[next_layer]
    T_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last_layer] - T[next_layer])) + T[next_layer]
    L_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last_layer] - L[next_layer])) + L[next_layer]
    M_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last_layer] - M[next_layer])) + M[next_layer]
    #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')


    RIni = 0.0
    h = -h #paso de integración

    

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos

    Tc = 1.8
    temp = []
    ERROR = []
    while Tc <=1.9:
        Tc += 0.0001
        i=-1

        while i<=1:
            i += 1
            Tres_primeras_capas_cen(i)
            #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
            #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')
        #
        # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
        #

        i = 2  # número de la última capa calculada
        Counter = []
        #print("-" * 78)
        #print("Algoritmo A.2.")
        #print("-" * 78)

        # Utilizamos las variables lógicas loop1, loop2 y loop3
        # para controlar cada uno de los bucle necesarios para
        # programar el algoritmo de la fase A.1. (envoltura radiativa)

        loop1 = True
        while loop1:
            #print("\n*Calculando capa número:", i+1, "\n")
            #print("Ejecutando paso 1")
            r[i+1] = RIni+(i+1) * h
            #print("Ejecutando paso 2 bis")
            step_2_bis(i)

            loop2 = True
            while loop2:
                #print("Ejecutando polítropo 1")

                politropo_1(i)

                #print("Ejecutando paso 3")
                step_3(i)


                #print("Ejecutando paso 7 bis")

                step_7_bis(i)
                diference = step_8(i)
                if diference < ERelMax:
                    loop2 = False
                else:
                    #print("Hacemos Tcal = Test e iteramos loop2")
                    Test[i+1] = Tcal[i+1]
                    #input('Press <CR> to continue...')
            #print("Ejecutando polítropo 2")
            politropo_2(i)
            #print("Ejecutando paso 6")
            step_6_bis(i)

            decision = r[i+1]
            if decision >= 2.053:
                loop1 = False
            else:
                P[i+1] = Pcal[i+1]
                T[i+1] = Tcal[i+1]
                L[i+1] = Lcal[i+1]
                M[i+1] = Mcal[i+1]
                #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
                # pasamos a la siguiente capa
                i += 1
                Counter.append(i)


        #print("-" * 78)
        #print("Hemos llegado a la zona convectiva.")
        #print("-" * 78)

        last = len(Counter) + 2
        prev = len(Counter) + 1
        r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
        P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
        T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
        L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
        M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
        #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
        #print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')

        err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
        err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
        err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
        err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
        err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
        ERROR.append(err_Tot)
        temp.append(Tc)
        #print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')
    print(f'{LTot:8.1f} {min(ERROR):8.2f} ')
    


# In[71]:


LTot = 28.5
while LTot <= 44:
    
    LTot += 1.5
    
    RTot = 11.6
    MTot = 5  #masa total
    RIni = 0.9*RTot #radio inicial
    num_capas = 101 #número de capas
    X = 0.8 #contenido de hidrogeno
    Y = 0.16 #contenido de helio
    Z = 1-Y-X #elementos pesados
    mu = 1/((2*X)+(3*Y/4)+(Z/2)) #peso molecular medio
    A1 = 1.9022*mu*MTot #cte para la temp de las primeras capas
    A2 = 10.645*np.sqrt(MTot/(mu*Z*LTot*(1+X))) #cte para la pres de las primeras capas
    Cp = 8.084*mu #cte para calcular f_i de las pres
    Ct = 0.01679*Z*(1+X)*mu**2 #cte para calcular f_i de la temp
    h = -RIni/(num_capas-1) #paso de integración
    ERelMax = 0.0001
    X1_PP = X
    X2_PP = X
    X1_CN = X
    X2_CN = Z / 3
    rho = MTot / (4/3) * np.pi * RTot**3

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos
    n=np.zeros(num_capas)

    novotny = np.ones(num_capas) * 2.5

    i=-1

    while i<=1:
        i += 1
        Tres_primeras_capas(i)
        #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')



    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #

    i = 2  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.1.")
    #print("-" * 78)
    Counter = []
    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+ (i+1)*h
        #print("Ejecutando paso 2")
        step_2(i)    
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                #print("Ejecutando paso 3")
                step_3(i)
                #print("Ejecutando paso 4")
                step_4(i)

                #print(Pest[i+1], Pcal[i+1])
                diference = step_5(i)
                if diference < ERelMax:
                    loop3 = False
                else:
                    #print("Hacemos Pest = Pcal e iteramos loop3")
                    Pest[i+1] = Pcal[i+1]
                    #input('Press <CR> to continue...')

            #print("Ejecutando paso 6")

            step_6(i)

            #print("Ejecutando paso 7")
            step_7(i)

            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
                #input('Press <CR> to continue...')

        #print("Ejecutando paso 9")
        step_9(i)

        decision = n[i+1]  
        if decision <= 2.5:
            loop1 = False
        else:
            # almacenamos los valores calculados y los mostramos
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]
            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}')   

            # pasamos a la siguiente capa
            i += 1
            Counter.append(i)



        #input('Press <CR> to continue...')


    #print("-" * 78)
    #print("Pasamos a la fase convectiva, algoritmo A.2.")
    #print("-" * 78)


    # Ya tenemos las tres primeras capas.   

    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #
    last_layer = len(Counter) + 2
    next_layer = len(Counter) + 3
    x = r[last_layer]
    #print(x)
    i = last_layer  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.2.")
    #print("-" * 78)

    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+(i+1) * h
        step_9_bis(i)
        #print("Ejecutando paso 2 bis")
        step_2_bis(i)

        loop2 = True
        while loop2:
            #print("Ejecutando polítropo 1")

            politropo_1(i)

            #print("Ejecutando paso 3")
            step_3(i)


            #print("Ejecutando paso 7 bis")

            step_7_bis(i)
            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
        #print("Ejecutando polítropo 2")
        politropo_2(i)

        #print("Ejecutando paso 6")
        step_6_bis(i)


        decision = r[i+1]

        if decision <= 0:
            loop1 = False
        else:
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]

            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {n[i+1]:10.6f}') 
            # pasamos a la siguiente capa
            i += 1

    #print("-" * 78)
    #print("Hemos llegado al nucleo de la estrella.")
    #print("-" * 78)

    r_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last_layer] - r[next_layer])) + r[next_layer]
    P_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last_layer] - P[next_layer])) + P[next_layer]
    T_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last_layer] - T[next_layer])) + T[next_layer]
    L_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last_layer] - L[next_layer])) + L[next_layer]
    M_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last_layer] - M[next_layer])) + M[next_layer]
    #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')


    RIni = 0.0
    h = -h #paso de integración

    

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos

    Tc = 1.8
    temp = []
    ERROR = []
    while Tc <=1.9:
        Tc += 0.0001
        i=-1

        while i<=1:
            i += 1
            Tres_primeras_capas_cen(i)
            #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
            #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')
        #
        # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
        #

        i = 2  # número de la última capa calculada
        Counter = []
        #print("-" * 78)
        #print("Algoritmo A.2.")
        #print("-" * 78)

        # Utilizamos las variables lógicas loop1, loop2 y loop3
        # para controlar cada uno de los bucle necesarios para
        # programar el algoritmo de la fase A.1. (envoltura radiativa)

        loop1 = True
        while loop1:
            #print("\n*Calculando capa número:", i+1, "\n")
            #print("Ejecutando paso 1")
            r[i+1] = RIni+(i+1) * h
            #print("Ejecutando paso 2 bis")
            step_2_bis(i)

            loop2 = True
            while loop2:
                #print("Ejecutando polítropo 1")

                politropo_1(i)

                #print("Ejecutando paso 3")
                step_3(i)


                #print("Ejecutando paso 7 bis")

                step_7_bis(i)
                diference = step_8(i)
                if diference < ERelMax:
                    loop2 = False
                else:
                    #print("Hacemos Tcal = Test e iteramos loop2")
                    Test[i+1] = Tcal[i+1]
                    #input('Press <CR> to continue...')
            #print("Ejecutando polítropo 2")
            politropo_2(i)
            #print("Ejecutando paso 6")
            step_6_bis(i)

            decision = r[i+1]
            if decision >= 2.053:
                loop1 = False
            else:
                P[i+1] = Pcal[i+1]
                T[i+1] = Tcal[i+1]
                L[i+1] = Lcal[i+1]
                M[i+1] = Mcal[i+1]
                #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
                # pasamos a la siguiente capa
                i += 1
                Counter.append(i)


        #print("-" * 78)
        #print("Hemos llegado a la zona convectiva.")
        #print("-" * 78)

        last = len(Counter) + 2
        prev = len(Counter) + 1
        r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
        P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
        T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
        L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
        M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
        #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
        #print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')

        err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
        err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
        err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
        err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
        err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
        ERROR.append(err_Tot)
        temp.append(Tc)
        #print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')
    print(f'{LTot:8.1f} {min(ERROR):8.2f} ')
    


# In[72]:


LTot = 28.5
while LTot <= 44:
    
    LTot += 1.5
    
    RTot = 11.7
    MTot = 5  #masa total
    RIni = 0.9*RTot #radio inicial
    num_capas = 101 #número de capas
    X = 0.8 #contenido de hidrogeno
    Y = 0.16 #contenido de helio
    Z = 1-Y-X #elementos pesados
    mu = 1/((2*X)+(3*Y/4)+(Z/2)) #peso molecular medio
    A1 = 1.9022*mu*MTot #cte para la temp de las primeras capas
    A2 = 10.645*np.sqrt(MTot/(mu*Z*LTot*(1+X))) #cte para la pres de las primeras capas
    Cp = 8.084*mu #cte para calcular f_i de las pres
    Ct = 0.01679*Z*(1+X)*mu**2 #cte para calcular f_i de la temp
    h = -RIni/(num_capas-1) #paso de integración
    ERelMax = 0.0001
    X1_PP = X
    X2_PP = X
    X1_CN = X
    X2_CN = Z / 3
    rho = MTot / (4/3) * np.pi * RTot**3

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos
    n=np.zeros(num_capas)

    novotny = np.ones(num_capas) * 2.5

    i=-1

    while i<=1:
        i += 1
        Tres_primeras_capas(i)
        #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')



    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #

    i = 2  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.1.")
    #print("-" * 78)
    Counter = []
    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+ (i+1)*h
        #print("Ejecutando paso 2")
        step_2(i)    
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                #print("Ejecutando paso 3")
                step_3(i)
                #print("Ejecutando paso 4")
                step_4(i)

                #print(Pest[i+1], Pcal[i+1])
                diference = step_5(i)
                if diference < ERelMax:
                    loop3 = False
                else:
                    #print("Hacemos Pest = Pcal e iteramos loop3")
                    Pest[i+1] = Pcal[i+1]
                    #input('Press <CR> to continue...')

            #print("Ejecutando paso 6")

            step_6(i)

            #print("Ejecutando paso 7")
            step_7(i)

            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
                #input('Press <CR> to continue...')

        #print("Ejecutando paso 9")
        step_9(i)

        decision = n[i+1]  
        if decision <= 2.5:
            loop1 = False
        else:
            # almacenamos los valores calculados y los mostramos
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]
            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}')   

            # pasamos a la siguiente capa
            i += 1
            Counter.append(i)



        #input('Press <CR> to continue...')


    #print("-" * 78)
    #print("Pasamos a la fase convectiva, algoritmo A.2.")
    #print("-" * 78)


    # Ya tenemos las tres primeras capas.   

    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #
    last_layer = len(Counter) + 2
    next_layer = len(Counter) + 3
    x = r[last_layer]
    #print(x)
    i = last_layer  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.2.")
    #print("-" * 78)

    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+(i+1) * h
        step_9_bis(i)
        #print("Ejecutando paso 2 bis")
        step_2_bis(i)

        loop2 = True
        while loop2:
            #print("Ejecutando polítropo 1")

            politropo_1(i)

            #print("Ejecutando paso 3")
            step_3(i)


            #print("Ejecutando paso 7 bis")

            step_7_bis(i)
            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
        #print("Ejecutando polítropo 2")
        politropo_2(i)

        #print("Ejecutando paso 6")
        step_6_bis(i)


        decision = r[i+1]

        if decision <= 0:
            loop1 = False
        else:
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]

            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {n[i+1]:10.6f}') 
            # pasamos a la siguiente capa
            i += 1

    #print("-" * 78)
    #print("Hemos llegado al nucleo de la estrella.")
    #print("-" * 78)

    r_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last_layer] - r[next_layer])) + r[next_layer]
    P_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last_layer] - P[next_layer])) + P[next_layer]
    T_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last_layer] - T[next_layer])) + T[next_layer]
    L_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last_layer] - L[next_layer])) + L[next_layer]
    M_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last_layer] - M[next_layer])) + M[next_layer]
    #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')


    RIni = 0.0
    h = -h #paso de integración

    

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos

    Tc = 1.8
    temp = []
    ERROR = []
    while Tc <=1.9:
        Tc += 0.0001
        i=-1

        while i<=1:
            i += 1
            Tres_primeras_capas_cen(i)
            #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
            #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')
        #
        # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
        #

        i = 2  # número de la última capa calculada
        Counter = []
        #print("-" * 78)
        #print("Algoritmo A.2.")
        #print("-" * 78)

        # Utilizamos las variables lógicas loop1, loop2 y loop3
        # para controlar cada uno de los bucle necesarios para
        # programar el algoritmo de la fase A.1. (envoltura radiativa)

        loop1 = True
        while loop1:
            #print("\n*Calculando capa número:", i+1, "\n")
            #print("Ejecutando paso 1")
            r[i+1] = RIni+(i+1) * h
            #print("Ejecutando paso 2 bis")
            step_2_bis(i)

            loop2 = True
            while loop2:
                #print("Ejecutando polítropo 1")

                politropo_1(i)

                #print("Ejecutando paso 3")
                step_3(i)


                #print("Ejecutando paso 7 bis")

                step_7_bis(i)
                diference = step_8(i)
                if diference < ERelMax:
                    loop2 = False
                else:
                    #print("Hacemos Tcal = Test e iteramos loop2")
                    Test[i+1] = Tcal[i+1]
                    #input('Press <CR> to continue...')
            #print("Ejecutando polítropo 2")
            politropo_2(i)
            #print("Ejecutando paso 6")
            step_6_bis(i)

            decision = r[i+1]
            if decision >= 2.053:
                loop1 = False
            else:
                P[i+1] = Pcal[i+1]
                T[i+1] = Tcal[i+1]
                L[i+1] = Lcal[i+1]
                M[i+1] = Mcal[i+1]
                #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
                # pasamos a la siguiente capa
                i += 1
                Counter.append(i)


        #print("-" * 78)
        #print("Hemos llegado a la zona convectiva.")
        #print("-" * 78)

        last = len(Counter) + 2
        prev = len(Counter) + 1
        r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
        P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
        T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
        L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
        M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
        #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
        #print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')

        err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
        err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
        err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
        err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
        err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
        ERROR.append(err_Tot)
        temp.append(Tc)
        #print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')
    print(f'{LTot:8.1f} {min(ERROR):8.2f} ')
    


# In[73]:


LTot = 28.5
while LTot <= 44:
    
    LTot += 1.5
    
    RTot = 11.8
    MTot = 5  #masa total
    RIni = 0.9*RTot #radio inicial
    num_capas = 101 #número de capas
    X = 0.8 #contenido de hidrogeno
    Y = 0.16 #contenido de helio
    Z = 1-Y-X #elementos pesados
    mu = 1/((2*X)+(3*Y/4)+(Z/2)) #peso molecular medio
    A1 = 1.9022*mu*MTot #cte para la temp de las primeras capas
    A2 = 10.645*np.sqrt(MTot/(mu*Z*LTot*(1+X))) #cte para la pres de las primeras capas
    Cp = 8.084*mu #cte para calcular f_i de las pres
    Ct = 0.01679*Z*(1+X)*mu**2 #cte para calcular f_i de la temp
    h = -RIni/(num_capas-1) #paso de integración
    ERelMax = 0.0001
    X1_PP = X
    X2_PP = X
    X1_CN = X
    X2_CN = Z / 3
    rho = MTot / (4/3) * np.pi * RTot**3

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos
    n=np.zeros(num_capas)

    novotny = np.ones(num_capas) * 2.5

    i=-1

    while i<=1:
        i += 1
        Tres_primeras_capas(i)
        #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')



    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #

    i = 2  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.1.")
    #print("-" * 78)
    Counter = []
    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+ (i+1)*h
        #print("Ejecutando paso 2")
        step_2(i)    
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                #print("Ejecutando paso 3")
                step_3(i)
                #print("Ejecutando paso 4")
                step_4(i)

                #print(Pest[i+1], Pcal[i+1])
                diference = step_5(i)
                if diference < ERelMax:
                    loop3 = False
                else:
                    #print("Hacemos Pest = Pcal e iteramos loop3")
                    Pest[i+1] = Pcal[i+1]
                    #input('Press <CR> to continue...')

            #print("Ejecutando paso 6")

            step_6(i)

            #print("Ejecutando paso 7")
            step_7(i)

            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
                #input('Press <CR> to continue...')

        #print("Ejecutando paso 9")
        step_9(i)

        decision = n[i+1]  
        if decision <= 2.5:
            loop1 = False
        else:
            # almacenamos los valores calculados y los mostramos
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]
            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}')   

            # pasamos a la siguiente capa
            i += 1
            Counter.append(i)



        #input('Press <CR> to continue...')


    #print("-" * 78)
    #print("Pasamos a la fase convectiva, algoritmo A.2.")
    #print("-" * 78)


    # Ya tenemos las tres primeras capas.   

    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #
    last_layer = len(Counter) + 2
    next_layer = len(Counter) + 3
    x = r[last_layer]
    #print(x)
    i = last_layer  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.2.")
    #print("-" * 78)

    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+(i+1) * h
        step_9_bis(i)
        #print("Ejecutando paso 2 bis")
        step_2_bis(i)

        loop2 = True
        while loop2:
            #print("Ejecutando polítropo 1")

            politropo_1(i)

            #print("Ejecutando paso 3")
            step_3(i)


            #print("Ejecutando paso 7 bis")

            step_7_bis(i)
            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
        #print("Ejecutando polítropo 2")
        politropo_2(i)

        #print("Ejecutando paso 6")
        step_6_bis(i)


        decision = r[i+1]

        if decision <= 0:
            loop1 = False
        else:
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]

            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {n[i+1]:10.6f}') 
            # pasamos a la siguiente capa
            i += 1

    #print("-" * 78)
    #print("Hemos llegado al nucleo de la estrella.")
    #print("-" * 78)

    r_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last_layer] - r[next_layer])) + r[next_layer]
    P_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last_layer] - P[next_layer])) + P[next_layer]
    T_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last_layer] - T[next_layer])) + T[next_layer]
    L_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last_layer] - L[next_layer])) + L[next_layer]
    M_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last_layer] - M[next_layer])) + M[next_layer]
    #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')


    RIni = 0.0
    h = -h #paso de integración

    

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos

    Tc = 1.8
    temp = []
    ERROR = []
    while Tc <=1.9:
        Tc += 0.0001
        i=-1

        while i<=1:
            i += 1
            Tres_primeras_capas_cen(i)
            #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
            #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')
        #
        # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
        #

        i = 2  # número de la última capa calculada
        Counter = []
        #print("-" * 78)
        #print("Algoritmo A.2.")
        #print("-" * 78)

        # Utilizamos las variables lógicas loop1, loop2 y loop3
        # para controlar cada uno de los bucle necesarios para
        # programar el algoritmo de la fase A.1. (envoltura radiativa)

        loop1 = True
        while loop1:
            #print("\n*Calculando capa número:", i+1, "\n")
            #print("Ejecutando paso 1")
            r[i+1] = RIni+(i+1) * h
            #print("Ejecutando paso 2 bis")
            step_2_bis(i)

            loop2 = True
            while loop2:
                #print("Ejecutando polítropo 1")

                politropo_1(i)

                #print("Ejecutando paso 3")
                step_3(i)


                #print("Ejecutando paso 7 bis")

                step_7_bis(i)
                diference = step_8(i)
                if diference < ERelMax:
                    loop2 = False
                else:
                    #print("Hacemos Tcal = Test e iteramos loop2")
                    Test[i+1] = Tcal[i+1]
                    #input('Press <CR> to continue...')
            #print("Ejecutando polítropo 2")
            politropo_2(i)
            #print("Ejecutando paso 6")
            step_6_bis(i)

            decision = r[i+1]
            if decision >= 2.053:
                loop1 = False
            else:
                P[i+1] = Pcal[i+1]
                T[i+1] = Tcal[i+1]
                L[i+1] = Lcal[i+1]
                M[i+1] = Mcal[i+1]
                #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
                # pasamos a la siguiente capa
                i += 1
                Counter.append(i)


        #print("-" * 78)
        #print("Hemos llegado a la zona convectiva.")
        #print("-" * 78)

        last = len(Counter) + 2
        prev = len(Counter) + 1
        r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
        P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
        T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
        L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
        M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
        #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
        #print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')

        err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
        err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
        err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
        err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
        err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
        ERROR.append(err_Tot)
        temp.append(Tc)
        #print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')
    print(f'{LTot:8.1f} {min(ERROR):8.2f} ')
    


# In[74]:


LTot = 28.5
while LTot <= 44:
    
    LTot += 1.5
    
    RTot = 11.9
    MTot = 5  #masa total
    RIni = 0.9*RTot #radio inicial
    num_capas = 101 #número de capas
    X = 0.8 #contenido de hidrogeno
    Y = 0.16 #contenido de helio
    Z = 1-Y-X #elementos pesados
    mu = 1/((2*X)+(3*Y/4)+(Z/2)) #peso molecular medio
    A1 = 1.9022*mu*MTot #cte para la temp de las primeras capas
    A2 = 10.645*np.sqrt(MTot/(mu*Z*LTot*(1+X))) #cte para la pres de las primeras capas
    Cp = 8.084*mu #cte para calcular f_i de las pres
    Ct = 0.01679*Z*(1+X)*mu**2 #cte para calcular f_i de la temp
    h = -RIni/(num_capas-1) #paso de integración
    ERelMax = 0.0001
    X1_PP = X
    X2_PP = X
    X1_CN = X
    X2_CN = Z / 3
    rho = MTot / (4/3) * np.pi * RTot**3

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos
    n=np.zeros(num_capas)

    novotny = np.ones(num_capas) * 2.5

    i=-1

    while i<=1:
        i += 1
        Tres_primeras_capas(i)
        #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')



    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #

    i = 2  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.1.")
    #print("-" * 78)
    Counter = []
    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+ (i+1)*h
        #print("Ejecutando paso 2")
        step_2(i)    
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                #print("Ejecutando paso 3")
                step_3(i)
                #print("Ejecutando paso 4")
                step_4(i)

                #print(Pest[i+1], Pcal[i+1])
                diference = step_5(i)
                if diference < ERelMax:
                    loop3 = False
                else:
                    #print("Hacemos Pest = Pcal e iteramos loop3")
                    Pest[i+1] = Pcal[i+1]
                    #input('Press <CR> to continue...')

            #print("Ejecutando paso 6")

            step_6(i)

            #print("Ejecutando paso 7")
            step_7(i)

            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
                #input('Press <CR> to continue...')

        #print("Ejecutando paso 9")
        step_9(i)

        decision = n[i+1]  
        if decision <= 2.5:
            loop1 = False
        else:
            # almacenamos los valores calculados y los mostramos
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]
            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}')   

            # pasamos a la siguiente capa
            i += 1
            Counter.append(i)



        #input('Press <CR> to continue...')


    #print("-" * 78)
    #print("Pasamos a la fase convectiva, algoritmo A.2.")
    #print("-" * 78)


    # Ya tenemos las tres primeras capas.   

    #
    # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
    #
    last_layer = len(Counter) + 2
    next_layer = len(Counter) + 3
    x = r[last_layer]
    #print(x)
    i = last_layer  # número de la última capa calculada

    #print("-" * 78)
    #print("Algoritmo A.2.")
    #print("-" * 78)

    # Utilizamos las variables lógicas loop1, loop2 y loop3
    # para controlar cada uno de los bucle necesarios para
    # programar el algoritmo de la fase A.1. (envoltura radiativa)

    loop1 = True
    while loop1:
        #print("\n*Calculando capa número:", i+1, "\n")
        #print("Ejecutando paso 1")
        r[i+1] = RIni+(i+1) * h
        step_9_bis(i)
        #print("Ejecutando paso 2 bis")
        step_2_bis(i)

        loop2 = True
        while loop2:
            #print("Ejecutando polítropo 1")

            politropo_1(i)

            #print("Ejecutando paso 3")
            step_3(i)


            #print("Ejecutando paso 7 bis")

            step_7_bis(i)
            diference = step_8(i)
            if diference < ERelMax:
                loop2 = False
            else:
                #print("Hacemos Tcal = Test e iteramos loop2")
                Test[i+1] = Tcal[i+1]
        #print("Ejecutando polítropo 2")
        politropo_2(i)

        #print("Ejecutando paso 6")
        step_6_bis(i)


        decision = r[i+1]

        if decision <= 0:
            loop1 = False
        else:
            P[i+1] = Pcal[i+1]
            T[i+1] = Tcal[i+1]
            L[i+1] = Lcal[i+1]
            M[i+1] = Mcal[i+1]

            #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {n[i+1]:10.6f}') 
            # pasamos a la siguiente capa
            i += 1

    #print("-" * 78)
    #print("Hemos llegado al nucleo de la estrella.")
    #print("-" * 78)

    r_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last_layer] - r[next_layer])) + r[next_layer]
    P_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last_layer] - P[next_layer])) + P[next_layer]
    T_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last_layer] - T[next_layer])) + T[next_layer]
    L_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last_layer] - L[next_layer])) + L[next_layer]
    M_int_d = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last_layer] - M[next_layer])) + M[next_layer]
    #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')


    RIni = 0.0
    h = -h #paso de integración

    

    T=np.zeros(num_capas)
    Test=np.zeros(num_capas)
    Tcal=np.zeros(num_capas)

    P=np.zeros(num_capas)
    Pest=np.zeros(num_capas)
    Pcal=np.zeros(num_capas)

    fp=np.zeros(num_capas)
    ft=np.zeros(num_capas)
    fm=np.zeros(num_capas)
    fl=np.zeros(num_capas)


    M=np.zeros(num_capas)
    Mcal=np.zeros(num_capas)
    Mest=np.zeros(num_capas)

    L=np.zeros(num_capas)
    Lcal=np.zeros(num_capas)
    GenEnergiaPP=np.zeros(num_capas)
    GenEnergiaCN=np.zeros(num_capas)

    r = np.zeros(num_capas) #arrays vacios para almacenar los datos

    Tc = 1.8
    temp = []
    ERROR = []
    while Tc <=1.9:
        Tc += 0.0001
        i=-1

        while i<=1:
            i += 1
            Tres_primeras_capas_cen(i)
            #print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
            #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')
        #
        # Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
        #

        i = 2  # número de la última capa calculada
        Counter = []
        #print("-" * 78)
        #print("Algoritmo A.2.")
        #print("-" * 78)

        # Utilizamos las variables lógicas loop1, loop2 y loop3
        # para controlar cada uno de los bucle necesarios para
        # programar el algoritmo de la fase A.1. (envoltura radiativa)

        loop1 = True
        while loop1:
            #print("\n*Calculando capa número:", i+1, "\n")
            #print("Ejecutando paso 1")
            r[i+1] = RIni+(i+1) * h
            #print("Ejecutando paso 2 bis")
            step_2_bis(i)

            loop2 = True
            while loop2:
                #print("Ejecutando polítropo 1")

                politropo_1(i)

                #print("Ejecutando paso 3")
                step_3(i)


                #print("Ejecutando paso 7 bis")

                step_7_bis(i)
                diference = step_8(i)
                if diference < ERelMax:
                    loop2 = False
                else:
                    #print("Hacemos Tcal = Test e iteramos loop2")
                    Test[i+1] = Tcal[i+1]
                    #input('Press <CR> to continue...')
            #print("Ejecutando polítropo 2")
            politropo_2(i)
            #print("Ejecutando paso 6")
            step_6_bis(i)

            decision = r[i+1]
            if decision >= 2.053:
                loop1 = False
            else:
                P[i+1] = Pcal[i+1]
                T[i+1] = Tcal[i+1]
                L[i+1] = Lcal[i+1]
                M[i+1] = Mcal[i+1]
                #print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
                # pasamos a la siguiente capa
                i += 1
                Counter.append(i)


        #print("-" * 78)
        #print("Hemos llegado a la zona convectiva.")
        #print("-" * 78)

        last = len(Counter) + 2
        prev = len(Counter) + 1
        r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
        P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
        T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
        L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
        M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
        #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
        #print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')

        err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
        err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
        err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
        err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
        err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
        ERROR.append(err_Tot)
        temp.append(Tc)
        #print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')
    print(f'{LTot:8.1f} {min(ERROR):8.2f} ')
    


# In[77]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[78]:


X= np.array([[19.32,19.36,19.93,16.06,13.7,9.33,5.79,7.15,15.01,9.64,3.99,2.51,30.5,25.15,19.05,34.77],
            [21.68,21.65,21.49,17.7,15.5,10.35,7.69,9.37,27.76,22.72,17.47,12.09,30.02,24.84,19.06,36.12],
            [24.15,23.99,23.35,19.57,17.45,23.79,18.77,14.26,11.38,8.71,4.05,5.76,16.31,10.81,4.92,36.67],
            [28.95,33.12,25.45,21.6,19.52,14.16,12.43,14.14,18.74,9.3,5.7,7.82,15.45,9.72,3.71,24.42],
            [29.25,28.84,22.51,23.27,26.53,26.57,21.32,16.94,14.58,10.54,7.77,10.01,29.2,23.69,17.94,36.73],
            [31.85,31.31,25.52,26.09,29.22,19.21,17.66,19.21,23.54,24.17,18.34,12.98,29.31,23.53,17.53,22.94],
            [34.46,33.81,32.64,28.32,26.13,30.3,24.7,20.34,18.86,14.31,12.44,14.68,15.28,9.25,4.89,22.45],
            [37.08,36.34,35.2,30.67,28.42,32.41,26.59,22.21,21.01,16.56,14.89,17.12,16.13,10.27,6.9,22.26],
            [39.68,38.89,37.8,33.05,30.76,34.63,28.58,24.16,34.6,27.76,21.46,16.44,30.97,24.21,17.46,36.99],
            [42.49,41.46,40.42,35.44,33.12,36.92,30.62,26.17,36.46,29.29,22.83,17.96,31.92,24.81,17.85,22.81]])

Y=np.array([[0,1,2],
           [1,2,3],
          [1,2,3]])


# In[79]:



fig,ax = plt.subplots()
ax.imshow(X)
ax.set_aspect('equal')


# In[ ]:




