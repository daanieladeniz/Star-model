#!/usr/bin/env python
# coding: utf-8

# # Programación de ecuaciones
# En este jupyter notebook, vamos a programar las ecuaciones para resolver un modelo numérico del interior estelar

# # Cálculo para las tres primeras capas

# ## Constantes

# In[1]:


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
ERelMax = 0.0001 # error relativo máximo utilizado en las decisiones

#Concentraciones para ciclos PP y CN
X1_PP = X
X2_PP = X
X1_CN = X
X2_CN = Z / 3

#Variables vacías para ir rellenando
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

r = np.zeros(num_capas) 
n=np.zeros(num_capas)


# In[2]:


# Parámetro para imprimir resultados y detectar errores
DEBUG = False
#DEBUG = True


# # Envoltura radiativa

# ## Definiciones para las tres primeras capas

# In[3]:


def T_primeras_capas(A1, r, RTot):
    return A1*((1/r)-(1/RTot))

def P_primeras_capas(A2, T):
    return A2*T**(4.25)

def f_i_presion(Cp, P, T, MTot, r):
    return (-Cp*P*MTot)/(T*r**2)

def f_i_temperatura(Ct, P, LTot, T, r):
    return (-Ct*P**2*LTot)/(T**8.5*r**2)


# ## Bucle para las tres primeras capas

# In[4]:


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


# In[5]:


i=-1
P_final = []
T_final = []
L_final = []
M_final = []
while i<=1:
    i += 1
    Tres_primeras_capas(i)
    P_final.append(P[i])
    T_final.append(T[i])
    L_final.append(L[i])
    M_final.append(M[i])
    
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
    
# Ya tenemos las tres primeras capas.              


# ## Definiciones para capas consecutivas

# In[6]:


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


# ## Bucle iterativo para capas consecutivas

# In[7]:


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
        P_final.append(P[i])
        T_final.append(T[i+1])
        L_final.append(L[i+1])
        M_final.append(M[i+1])
        print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}')   

        # pasamos a la siguiente capa
        
        i += 1
        
        Counter.append(i)
        
    
    #input('Press <CR> to continue...')


print("-" * 78)
print("Pasamos a la fase convectiva, algoritmo A.2.")
print("-" * 78)


# In[8]:


# Definición de la última capa radiativ, primera convectiva y el radio de la última capa radiativa
last_layer = len(Counter) + 2
next_layer = len(Counter) + 3
x = r[last_layer]


# # Nulceo convectivo

# ## Definiciones para el núcleo convectivo

# In[9]:


k = Pest[i+1] / (Test[i+1])**2.5

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
    
    
    
    
    

        


# ## Bucle iterativo para el núcleo convectivo

# In[10]:


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

# In[11]:


# Empleamos la fórmula y=((x-x1)/(x2-x1)*(y2-y1))+ y1


# In[12]:


r_int_d = ((2.5 - n[82]) / (n[81] - n[82]) * (r[81] - r[82])) + r[82]
P_int_d = ((2.5 - n[82]) / (n[81] - n[82]) * (P[81] - P[82])) + P[82]
T_int_d = ((2.5 - n[82]) / (n[81] - n[82]) * (T[81] - T[82])) + T[82]
L_int_d = ((2.5 - n[82]) / (n[81] - n[82]) * (L[81] - L[82])) + L[82]
M_int_d = ((2.5 - n[82]) / (n[81] - n[82]) * (M[81] - M[82])) + M[82]
print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')


# In[13]:


r_int_d = ((2.5 - n[last_layer]) / (n[next_layer] - n[last_layer]) * (r[next_layer] - r[last_layer])) + r[last_layer]
P_int_d = ((2.5 - n[last_layer]) / (n[next_layer] - n[last_layer]) * (P[next_layer] - P[last_layer])) + P[last_layer]
T_int_d = ((2.5 - n[last_layer]) / (n[next_layer] - n[last_layer]) * (T[next_layer] - T[last_layer])) + T[last_layer]
L_int_d = ((2.5 - n[last_layer]) / (n[next_layer] - n[last_layer]) * (L[next_layer] - L[last_layer])) + L[last_layer]
M_int_d = ((2.5 - n[last_layer]) / (n[next_layer] - n[last_layer]) * (M[next_layer] - M[last_layer])) + M[last_layer]
print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')


# # Integración desde el centro

# ## Definiciones para las primeras tres capas

# In[14]:


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


# In[15]:


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


# In[16]:


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
    
    
    


# ## Bucle para las tres primeras capas

# In[17]:


i=-1

while i<=1:
    i += 1
    Tres_primeras_capas_cen(i)
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
    #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')


# ## Iteración para capas posteriores

# In[18]:


#
# Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
#
P_c = []
T_c = []
L_c = []
M_c = []
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
        P_c.append(P[i+1])
        T_c.append(T[i+1])
        L_c.append(L[i+1])
        M_c.append(M[i+1])
        print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
        # pasamos a la siguiente capa
        i += 1
        Counter.append(i)

print("-" * 78)
print("Hemos llegado a la zona radiativa.")
print("-" * 78)


# ## Representación de los resultados

# In[19]:


idx = len(P_c) - 1
newP = []
newL = []
newT = []
newM = []
while (idx >= 0):
    newP.append(P_c[idx])
    newL.append(L_c[idx])
    newT.append(T_c[idx])
    newM.append(M_c[idx])
    idx = idx - 1

#print(newM)
for i in newP:
    P_final.append(i)


for i in newT:
    T_final.append(i)
    
for i in newL:
    L_final.append(i)
    
for i in newM:
    M_final.append(i)

x = [43.9452,44.15089,44.21955]
for i in x:
    P_final.append(i)
    
x= [1.49627,1.49906,1.50000]
for i in x:
    T_final.append(i)
    
x=[0.005505,0.000688,0.000]
for i in x:
    L_final.append(i)

x=[0.000867,0.000108,0.000]

for i in x:
    M_final.append(i)
    


# In[20]:


P_plot=np.array(P_final)/max(P_final)
T_plot=np.array(T_final)/max(T_final)
L_plot=np.array(L_final)/max(L_final)
M_plot=np.array(M_final)/max(M_final)


# In[21]:


sns.set_style('whitegrid')
plt.plot(P_plot,color='red',label='Presión')
plt.plot(T_plot,color='green',label='Temperatura')
plt.plot(L_plot,color='blue',label='Luminosidad')
plt.plot(M_plot,color='yellow',label='Masa')
plt.xlabel('Capas(i)')
plt.ylabel('Parámetros normalizados')
plt.title('Modelo completo para condiciones iniciales')
plt.legend()
plt.savefig('Completo_inicial.pdf')
plt.show()


# ##  Interpolación 

# In[22]:



last = len(Counter) + 2
prev = len(Counter) + 1

r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')


# ## Cálculo del error relativo total

# In[23]:


err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')

#Como la luminosidad integrando desde el centro sale muy bajo, podemos usar una Tc más alta para reducir el error


# ## Temperatura crítica $T_c=1.7$

# In[24]:


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




# In[25]:


i=-1

while i<=1:
    i += 1
    Tres_primeras_capas_cen(i)
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
    #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')


# In[26]:


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


# In[27]:


last = len(Counter) + 2
prev = len(Counter) + 1
r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
    #print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
    #print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')


# In[28]:


err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')


# ## Temperatura crítica $T_c=1.85$

# In[29]:


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


# In[30]:



i=-1

while i<=1:
    i += 1
    Tres_primeras_capas_cen(i)
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
    #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')


# In[31]:


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
#print("Hemos llegado a la zona radiativa.")
#print("-" * 78)


# In[32]:


last = len(Counter) + 2
prev = len(Counter) + 1


# In[33]:


r_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (r[last] - r[prev])) + r[prev]
P_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (P[last] - P[prev])) + P[prev]
T_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (T[last] - T[prev])) + T[prev]
L_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (L[last] - L[prev])) + L[prev]
M_int_u = ((2.5 - n[next_layer]) / (n[last_layer] - n[next_layer]) * (M[last] - M[prev])) + M[prev]
print(f' {r_int_d:8.5f} {P_int_d:10.4f} {T_int_d:10.4f} {L_int_d:10.3f} {M_int_d:10.4f} ')
print(f' {r_int_u:8.5f} {P_int_u:10.4f} {T_int_u:10.4f} {L_int_u:10.3f} {M_int_u:10.4f} ')


# In[34]:


err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')


# ## Bucle para $T_c$ entre 1.8 y 1.9

# In[35]:


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


# In[36]:


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
    


# In[37]:



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
        
        


# ## Representación del error en función de la temperatura

# In[38]:


min_err = min(ERROR)
index = ERROR.index(min(ERROR))
min_temp = temp[index]
print('--- El error mínimo es:',min_err)
print('--- La temperatura central es:',min_temp)
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

# In[39]:


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


# In[40]:


i=-1
P_final = []
T_final = []
L_final = []
M_final = []
while i<=9:
    i += 1
    
    Tres_primeras_capas(i)
    P_final.append(P[i])
    T_final.append(T[i])
    L_final.append(L[i])
    M_final.append(M[i])
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')


# In[41]:


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


# In[42]:


i=-1

while i<=1:
    i += 1
    Tres_primeras_capas(i)
    P_final.append(P[i])
    T_final.append(T[i])
    L_final.append(L[i])
    M_final.append(M[i])
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')

# Ya tenemos las tres primeras capas.    


# In[43]:


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
        P_final.append(P[i+1])
        T_final.append(T[i+1])
        L_final.append(L[i+1])
        M_final.append(M[i+1])
        print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}') 
        

        # pasamos a la siguiente capa
        i += 1
        
    
    #input('Press <CR> to continue...')


print("-" * 78)
print("Pasamos a la fase convectiva, algoritmo A.2.")
print("-" * 78)


# In[44]:


RIni = 0
h = 0.108 #paso de integración
Tc = 1.8561 #Temperatura en el centro

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


# In[45]:


i=-1

while i<=1:
    i += 1
    Tres_primeras_capas_cen(i)
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
    #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')
    


# In[46]:


#
# Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
#

P_c = []
T_c = []
L_c = []
M_c = []

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
        P_c.append(P[i+1])
        T_c.append(T[i+1])
        L_c.append(L[i+1])
        M_c.append(M[i+1])
        print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} ') 
        # pasamos a la siguiente capa
        i += 1

print("-" * 78)
print("Hemos llegado a la zona radiativa.")
print("-" * 78)


# #  Variación entre $L_{Tot}\in[30,45]$ y $R_{Tot}\in[11.0,11.9]$

# In[48]:


LTot = 28.5

while LTot <= 44 :
    
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



        


    
    last_layer = len(Counter) + 2
    next_layer = len(Counter) + 3
    x = r[last_layer] + 0.01
    
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
            if decision >= x:
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
        #print(last)
        prev = len(Counter) + 1
        #print(prev)
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
        index = ERROR.index(min(ERROR))
        min_temp = temp[index]
    #print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')
    print(f'{LTot:8.1f} {min(ERROR):8.2f} ')
    
    


# In[49]:


LTot = 28.5

while LTot <= 44 :
    
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
    x = r[last_layer] + 0.01
    
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
            if decision >= x:
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
        #print(last)
        prev = len(Counter) + 1
        #print(prev)
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
        index = ERROR.index(min(ERROR))
        min_temp = temp[index]
    #print(f' {err_P:8.5f} {err_T:10.4f} {err_L:10.4f} {err_M:10.3f} {err_Tot:10.3f} ')
    print(f'{LTot:8.1f} {min(ERROR):8.2f} ')
    
    


# In[81]:


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
    x = r[last_layer] + 0.01
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
            if decision >= x:
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
    


# In[82]:


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
    x = r[last_layer] + 0.01
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
            if decision >= x:
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
    print(f'{LTot:8.1f} {min(ERROR):8.2f}  ')
    


# In[83]:


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
    x = r[last_layer] + 0.01
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
            if decision >= x:
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

    


# In[84]:


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
    x = r[last_layer] + 0.01
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
            if decision >= x:
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
    


# In[85]:


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
    x = r[last_layer] + 0.01
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
            if decision >= x:
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
    


# In[86]:


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
    x = r[last_layer] + 0.01
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
            if decision >= x:
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
    


# In[87]:


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
    x = r[last_layer] + 0.01
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
            if decision >= x:
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
    


# In[88]:


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
    x = r[last_layer] + 0.01
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
            if decision >= x:
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
    


# # Zoom en zona de menor error

# In[89]:


LTot = 35
while LTot <= 41:
    
    LTot += 1
    
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
            if decision >= x:
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
    


# In[90]:


LTot = 35
while LTot <= 41:
    
    LTot += 1
    
    RTot = 11.05
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
            if decision >= x:
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
    


# In[59]:


LTot = 35
while LTot <= 41:
    
    LTot += 1
    
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
            if decision >= x:
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
        

        err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
        err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
        err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
        err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
        err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
        ERROR.append(err_Tot)
        temp.append(Tc)
        
    print(f'{LTot:8.1f} {min(ERROR):8.2f} ')
    


# In[91]:


LTot = 35
while LTot <= 41:
    
    LTot += 1
    
    RTot = 11.15
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
            if decision >= x:
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
        

        err_P = abs(((P_int_d - P_int_u) / P_int_d)*100)
        err_T = abs(((T_int_d - T_int_u) / T_int_d) *100)
        err_L = abs(((L_int_d - L_int_u) / L_int_d) *100)
        err_M = abs(((M_int_d - M_int_u) / M_int_d) *100)
        err_Tot = np.sqrt(err_P**2 + err_T**2 + err_L**2 + err_M**2)
        ERROR.append(err_Tot)
        temp.append(Tc)
        index = ERROR.index(min(ERROR))
        min_temp = temp[index]
        
    print(f'{LTot:8.1f} {min(ERROR):8.2f}  ')
    


# In[92]:


LTot = 35
while LTot <= 41:
    
    LTot += 1
    
    RTot = 11.20
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
    x = r[last_layer] + 0.01
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
            if decision >= x:
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


# In[35]:


LTot = 35
while LTot <= 41:
    
    LTot += 1
    
    RTot = 11.25
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
    x = r[last_layer] + 0.01
    #print(x)
    i = last_layer  # número de la última capa calculada
    k = Pest[i+1] / (Test[i+1])**2.5
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
            if decision >= x:
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


# In[94]:


LTot = 35
while LTot <= 41:
    
    LTot += 1
    
    RTot = 11.30
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
    x = r[last_layer] + 0.01
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
            if decision >= x:
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


# In[95]:


LTot = 35
while LTot <= 41:
    
    LTot += 1
    
    RTot = 11.35
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
    x = r[last_layer] + 0.01
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
            if decision >= x:
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


# In[96]:


LTot = 35
while LTot <= 41:
    
    LTot += 1
    
    RTot = 11.40
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
    x = r[last_layer] + 0.01
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
            if decision >= x:
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


# In[97]:


LTot = 35
while LTot <= 41:
    
    LTot += 1
    
    RTot = 11.45
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
    x = r[last_layer] + 0.01
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
            if decision >= x:
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


# # Color maps

# In[67]:


sns.set_style("white")

X= np.array([[22.16,24.82,27.54,30.28,33.03,35.8,38.57,41.33,44.07,46.81],
            [20.86,23.18,25.52,27.9,30.28,32.66,35.05,37.44,39.82,42.39],
            [13.6,16.18,18.83,21.51,24.23,26.95,29.68,32.4,35.12,37.82],
            [13.37,15.44,17.61,19.84,22.1,24.39,26.68,28.98,31.28,33.57],
            [5.73,8,10.5,13.11,15.78,18.48,21.2,23.91,26.63,29.33],
            [7.64,8.94,10.61,12.52,14.58,16.72,18.9,21.11,23.34,25.56],
            [5.26,3.91,4.11,5.78,8.08,10.62,13.27,15.96,18.68,21.4],
            [8.57,7.94,7.36,7.8,8.88,10.41,12.22,14.18,16.25,18.36],
            [12.87,10.6,8.42,6.58,5.53,5.78,7.25,9.38,11.82,14.41],
            [14.98,13.24,11.6,10.21,9.24,8.82,9.05,9.89,11.19,12.81],
            [20.2,18.37,16.3,14.13,11.99,10.12,8.77,8.26,8.76,10.13]])

fig, ax = plt.subplots()
shw = ax.imshow(X,cmap='viridis')
bar = plt.colorbar(shw)
font = {'fontsize':8}
ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax.set_xticklabels(['11', '11.1','11.2','11.3','11.4','11.5','11.6','11.7','11.8','11.9'],fontdict= font)
ax.set_yticks([0,1,2,3,4,5,6,7,8,9,10])
ax.set_yticklabels(['30.0','31.5','33.0','34.5','36.0','37.5','39.0','40.5','42.0','43.5','45.0'],fontdict= font)
label_font = {'fontsize':15}
ax.set_xlabel('$R_{Tot}$',fontdict=label_font)
ax.set_ylabel('$L_{Tot}$',fontdict=label_font)
plt.title('Color Map de errores relativos')

plt.savefig('ColorMap1.pdf')


# In[68]:


X= np.array([[5.73,6.83,8,9.23,10.5,11.79,13.11,14.44,15.78,17.12],
            [6.34,7.18,8.13,9.14,10.21,11.32,12.46,13.62,14.79,15.97],
            [5.93,5.1,4.57,10.69,11.34,12.06,12.85,13.68,14.56,15.47],
            [5.26,4.45,3.91,3.78,4.11,4.82,5.78,6.88,8.08,9.33],
            [7.06,6.47,6.06,5.87,5.93,6.24,6.77,7.48,8.31,9.25],
            [10.3,9.8,9.39,9.1,8.94,8.92,9.05,9.31,9.7,10.2],
            [12.87,11.74,10.6,9.49,8.42,7.43,6.58,5.91,5.53,5.48]])

sns.set_style("white")
fig, ax = plt.subplots()
shw = ax.imshow(X,cmap='viridis')
bar = plt.colorbar(shw)
font = {'fontsize':8}
ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax.set_xticklabels(['11.00', '11.05','11.1','11.15','11.20','11.25','11.30','11.35','11.40','11.45'],fontdict= font)
ax.set_yticks([0,1,2,3,4,5,6])
ax.set_yticklabels(['36','37','38','39','40','41','42'],fontdict= font)
label_font = {'fontsize':15}
ax.set_xlabel('$R_{Tot}$',fontdict=label_font)
ax.set_ylabel('$L_{Tot}$',fontdict=label_font)
plt.title('Color Map de errores relativos en zona más restringida')
plt.savefig('ColorMap2.pdf')


# ## Modelo completo para mejores datos encontrados
# ### $R_{Tot}=11.15$,$L_{Tot}=39$,$T_c=1.8544$

# In[22]:


MTot = 5  #masa total
LTot = 39 #luminosidd total
RTot = 11.15 #radio total
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

i=-1
P_final = []
T_final = []
L_final = []
M_final = []
while i<=1:
    i += 1
    Tres_primeras_capas(i)
    P_final.append(P[i])
    T_final.append(T[i])
    L_final.append(L[i])
    M_final.append(M[i])
    
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
    
    
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
        P_final.append(P[i])
        T_final.append(T[i+1])
        L_final.append(L[i+1])
        M_final.append(M[i+1])
        print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f} {decision:10.6f}')   

        # pasamos a la siguiente capa
        
        i += 1
        
        Counter.append(i)
        
    
    #input('Press <CR> to continue...')


print("-" * 78)
print("Pasamos a la fase convectiva, algoritmo A.2.")
print("-" * 78)






# In[23]:


last_layer = len(Counter) + 2
next_layer = len(Counter) + 3
x = r[last_layer]


# In[24]:


RIni = 0.0
h = -h #paso de integración
Tc = 1.8544 #Temperatura en el centro

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


i=-1

while i<=1:
    i += 1
    Tres_primeras_capas_cen(i)
    print(f'{i:3d} {r[i]:8.5f} {P[i]:10.7f} {T[i]:10.7f} {L[i]:10.6f} {M[i]:10.6f}')
    #print(f'{i:3d} {r[i]:8.5f} {fm[i]:10.7f} {fl[i]:10.7f} {ft[i]:10.6f} {fp[i]:10.6f}')
    
#
# Hemos calculado ya 3 capas iniciales (i = 0, 1 y 2)
#
P_c = []
T_c = []
L_c = []
M_c = []
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
    if decision  >= x:
        loop1 = False
    else:
        P[i+1] = Pcal[i+1]
        T[i+1] = Tcal[i+1]
        L[i+1] = Lcal[i+1]
        M[i+1] = Mcal[i+1]
        P_c.append(P[i+1])
        T_c.append(T[i+1])
        L_c.append(L[i+1])
        M_c.append(M[i+1])
        print(f'{i+1:3d} {r[i+1]:8.5f} {P[i+1]:10.7f} {T[i+1]:10.7f} {Lcal[i+1]:10.6f} {Mcal[i+1]:10.6f}') 
        # pasamos a la siguiente capa
        i += 1
        Counter.append(i)

print("-" * 78)
print("Hemos llegado a la zona radiativa.")
print("-" * 78)


# In[25]:


idx = len(P_c) - 1
newP = []
newL = []
newT = []
newM = []
while (idx >= 0):
    newP.append(P_c[idx])
    newL.append(L_c[idx])
    newT.append(T_c[idx])
    newM.append(M_c[idx])
    idx = idx - 1

#print(newM)
for i in newP:
    P_final.append(i)


for i in newT:
    T_final.append(i)
    
for i in newL:
    L_final.append(i)
    
for i in newM:
    M_final.append(i)


x=[0.005505,0.000688,0.000]
for i in x:
    L_final.append(i)

x=[0.000867,0.000108,0.000]

for i in x:
    M_final.append(i)
    


# In[26]:



P_plot=np.array(P_final)/max(P_final)
T_plot=np.array(T_final)/max(T_final)
L_plot=np.array(L_final)/max(L_final)
M_plot=np.array(M_final)/max(M_final)


# In[27]:


sns.set_style('whitegrid')
plt.plot(P_plot,color='red',label='Presión')
plt.plot(T_plot,color='green',label='Temperatura')
plt.plot(L_plot,color='blue',label='Luminosidad')
plt.plot(M_plot,color='yellow',label='Masa')
plt.xlabel('Capas(i)')
plt.ylabel('Parámetros normalizados')
plt.title('Modelo completo para condiciones finales')
plt.legend()
plt.savefig('Completo_final.pdf')
plt.show()


# ## Representación de nuestra estrella en el diagrama HR

# In[76]:


from hoki import load
import pandas as pd
sin_hr_diagram = load.model_output('/Users/daanieladeniz/opt/anaconda3/lib/python3.8/site-packages/hoki/data/hrs-sin-imf_chab100.zem4.dat', hr_type = 'TL')


# In[78]:



plt.figure(figsize=(8,6))


my_plot = sin_hr_diagram.plot(log_age=7.0)
my_plot.scatter(3.95,0.98, label='Nuestra estrella', s=80,color='red')



# Ajustamos los límites
my_plot.set_xlim([5.4, 3.4])
my_plot.set_ylim([0.3, 6.2]) 

my_plot.legend() # a legend
my_plot.set_title('Diagrama Hertzsprung-Russell') 
plt.savefig('HR.pdf')
plt.show()


# In[ ]:




