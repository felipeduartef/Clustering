# %% [markdown]
# #                                                   Tarea 2 - Algoritmos de Agrupamiento
# 
# ##                                                  Felipe Duarte Flórez - 1017240143
# 
# ###                                                      Ingeniería Electrónica
# ###                                              Fundamentos de Inteligencia Computacional

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from math import dist
import mpl_toolkits.mplot3d

# %%
iris = pd.read_excel("Datos IRIS.xlsx", index_col=None, header=None)
iris.plot(2, 3, kind="scatter", xlabel="X2", ylabel="X3")
plt.title('Descriptores seleccionados')
plt.grid()
plt.show()

# %% [markdown]
# ## División de los datos

# %%
# División de dataset

df_T = iris.iloc[0:35] # Datos de Entrenamiento
df_T = df_T.append(iris.iloc[50:85])
df_T = df_T.append(iris.iloc[100:135])

#print("Entrenamiento \n", df_T)


df_P = iris.iloc[35:50]          #Datos de prueba
df_P = df_P.append(iris.iloc[85:100])
df_P = df_P.append(iris.iloc[135:150])

#print("Prueba \n", df_P)

# %% [markdown]
# ## Normalización de los conjuntos de datos

# %%
#Mínimos y máximos
def minimo(dfin):
    min_0 = dfin[0].min()  #Columna 0
    min_1 = dfin[1].min()  #Columna 1
    min_2 = dfin[2].min()  #Columna 2
    min_3 = dfin[3].min()  #Columna 3

    return [min_0, min_1, min_2, min_3]

def maximo(dfin):
    max_0 = dfin[0].max()  #Columna 0
    max_1 = dfin[1].max()  #Columna 1
    max_2 = dfin[2].max()  #Columna 2
    max_3 = dfin[3].max()  #Columna 3

    return [max_0, max_1, max_2, max_3]

# Normalización por columna
def minmax_norm(dfin , min, max):
    return (dfin - min) / (max - min)

# Normalización del dataframe
def norm_df(dfin, min = [0,0,0,0], max = [0,0,0,0]):

    # Normalización de Dataframe de entrenamiento
    df_norm_0 = minmax_norm(dfin[0], min[0], max[0])  #Columna 0
    df_norm_1 = minmax_norm(dfin[1], min[1], max[1])  #Columna 1
    df_norm_2 = minmax_norm(dfin[2], min[2], max[2])  #Columna 2
    df_norm_3 = minmax_norm(dfin[3], min[3], max[3])  #Columna 3

    return pd.DataFrame([df_norm_0,df_norm_1, df_norm_2, df_norm_3, dfin[5]] ).transpose()  # Dataframe con las columnas normalizadas

#Mínimos y máximos para cada Dataframe
global_min = minimo(df_T)    # Mínimo global de los datos
global_max = maximo(df_T)    # Máximo global de los datos

minimo_P = minimo(df_P)
maximo_P = maximo(df_P)


#Actualización de mínimo y máximo global en caso de que sea necesario
for i in range(len(minimo_P)):
    if minimo_P[i] < global_min[i]:
        global_min[i] = minimo_P[i]
    if maximo_P[i] > global_max[i]:
        global_max[i] = maximo_P[i]

#print(global_min, global_max)

#Normalización Dataframe de entrenamiento
# Normalización de Dataframe de validación
df_norm_0 = minmax_norm(df_T[0], global_min[0], global_max[0])  #Columna 0
df_norm_1 = minmax_norm(df_T[1], global_min[1], global_max[1])  #Columna 1
df_norm_2 = minmax_norm(df_T[2], global_min[2], global_max[2])  #Columna 2
df_norm_3 = minmax_norm(df_T[3], global_min[3], global_max[3])  #Columna 3

df_TN = pd.DataFrame([df_norm_0,df_norm_1, df_norm_2, df_norm_3]).transpose()  # Dataframe con las columnas normalizadas

#print("Entrenamiento Normalizado \n", df_TN)

#Normalización Dataframe de prueba
df_PN = norm_df(df_P, global_min, global_max)
#print("Prueba Normalizado \n",df_PN)

# %%
# Calcular Error
etiquetas = df_T.iloc[:, -1].values
etiquetas_prueba = df_P.iloc[:, -1].values


def error(output, labels):
    count_error = 0      #contador de errores
    total_data = len(output)  #Total de datos

    for i in range(len(output)):
        if output[i] != labels[i]:
            count_error += 1
    
    percentage = count_error * 100/total_data           # Porcentaje de error
    percentage = round(percentage, 2)

    return percentage

# %% [markdown]
# ## Para 3 clusters

# %%
#Datos de entrenamiento normalizados con los dos descriptores seleccionados
x = df_TN.iloc[:, [2, 3]].values

centros = np.array([[0.1, 0.1], [0.3, 0.5], [0.6, 0.6]]) #Matriz que contiene las coordenadas de los centros

kmeans = KMeans(n_clusters = 3, init = centros, max_iter = 300, random_state = 0)     #Entrenamiento para hallar los centros
y_kmeans = kmeans.fit_predict(x)      #Obtener la clasificación de los datos de entrenamiento
print(y_kmeans)                       # Se muestra la clasificación resultante de los datos de entrenamiento


fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización delos grupos
ax1.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], c='green', marker='*', label = 'Iris-virginica')

# Visualización de los centroides de cada grupo
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("3 Grupos")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in y_kmeans:
    if i == 0:
        salida.append(1)
    elif i == 1:
        salida.append(2)
    else:
        salida.append(3)

print(salida)

print("Error de entrenamiento", error(salida, etiquetas))

# %%
###############     Prueba   ################

#Datos de prueba normalizadoscon los dos descriptores seleccionados
prueba = df_PN.iloc[:, [2, 3]].values

r_kmeans = kmeans.predict(prueba)    #Clasificación de los datos de prueba en base a los centros obtenidos
print(r_kmeans)                      #Se muestra el resultado de la clasificación

fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización delos grupos
ax1.scatter(prueba[r_kmeans == 0, 0], prueba[r_kmeans == 0, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(prueba[r_kmeans == 1, 0], prueba[r_kmeans == 1, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(prueba[r_kmeans == 2, 0], prueba[r_kmeans == 2, 1], c='green', marker='*', label = 'Iris-virginica')

# Visualización de los centroides de cada grupo
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("Prueba 3 Grupos")
ax1.legend()

plt.show()


#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in r_kmeans:
    if i == 0:
        salida.append(1)
    elif i == 1:
        salida.append(2)
    else:
        salida.append(3)

print(salida)

print(error(salida, etiquetas_prueba))


# %% [markdown]
# # Para 4 clusters

# %%
#Datos de entrenamiento normalizados con los dos descriptores seleccionados
x = df_TN.iloc[:, [2, 3]].values

centros = np.array([[0.1, 0.1], [0.2, 0.4], [0.6, 0.6], [0.8, 0.7]]) #Matriz que contiene las coordenadas de los centros

kmeans = KMeans(n_clusters = 4, init = centros , max_iter = 300, n_init = 10, random_state = 0) #Entrenamiento para hallar los centros
y_kmeans = kmeans.fit_predict(x)       #Obtener la clasificación de los datos de entrenamiento
print(y_kmeans)                        # Se muestra la clasificación resultante de los datos de entrenamiento


fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización delos grupos
ax1.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], c='blue', marker='o', label = 'Iris-versicolour')
ax1.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], c='green', marker='*', label = 'Iris-virginica')

# Visualización de los centros de cada grupo
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("4 Grupos")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in y_kmeans:
    if i == 0:
        salida.append(1)
    elif i == 2 or i == 1:
        salida.append(2)
    else:
        salida.append(3)

print(salida)
print("Error Entrenamiento:",error(salida, etiquetas), "%")

# %%
###############     Prueba   ################

#Datos de prueba normalizadoscon los dos descriptores seleccionados
prueba = df_PN.iloc[:, [2, 3]].values

r_kmeans = kmeans.predict(prueba)    #Clasificación de los datos de prueba en base a los centros obtenidos
print(r_kmeans)                      #Se muestra el resultado de la clasificación

fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización delos grupos
ax1.scatter(prueba[r_kmeans == 0, 0], prueba[r_kmeans == 0, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(prueba[r_kmeans == 2, 0], prueba[r_kmeans == 2, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(prueba[r_kmeans == 1, 0], prueba[r_kmeans == 1, 1], c='blue', marker='o', label = 'Iris-versicolour')
ax1.scatter(prueba[r_kmeans == 3, 0], prueba[r_kmeans == 3, 1], c='green', marker='*', label = 'Iris-virginica')

# Visualización de los centros de cada grupo
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("Prueba 4 Grupos")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in r_kmeans:
    if i == 0:
        salida.append(1)
    elif i == 2 or i == 1:
        salida.append(2)
    else:
        salida.append(3)

print(salida)
print(error(salida, etiquetas_prueba))


# %% [markdown]
# # Para 5 clusters

# %%
#Datos de entrenamiento normalizados con los dos descriptores seleccionados
x = df_TN.iloc[:, [2, 3]].values

centros = np.array([[0.1, 0.1], [0.2, 0.2], [0.4, 0.4], [0.6,0.6], [0.8,0.8]]) #Matriz que contiene las coordenadas de los centros

kmeans = KMeans(n_clusters = 5, init = centros, max_iter = 300, n_init = 10, random_state = 0)  #Entrenamiento para hallar los centros
y_kmeans = kmeans.fit_predict(x)            #Obtener la clasificación de los datos de entrenamiento
print(y_kmeans)                             # Se muestra la clasificación resultante de los datos de entrenamiento


fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización delos grupos
ax1.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], c='black', marker='+', label = 'Iris-versicolour')
ax1.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], c='green', marker='*', label = 'Iris-virginica')
ax1.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], c='blue', marker='o', label = 'Iris-virginica')

# Visualización de los centros de cada grupo
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("5 Grupos")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in y_kmeans:
    if i == 0 : 
        salida.append(1)
    elif i == 3 or i == 2:
        salida.append(2)
    else:
        salida.append(3)

print(salida)
print("Error de entrenamiento:",error(salida, etiquetas))

# %%
###############     Prueba   ################

#Datos de prueba normalizadoscon los dos descriptores seleccionados
prueba = df_PN.iloc[:, [2, 3]].values

r_kmeans = kmeans.predict(prueba)       #Clasificación de los datos de prueba en base a los centros obtenidos
print(r_kmeans)                         #Se muestra el resultado de la clasificación

fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización delos grupos
ax1.scatter(prueba[r_kmeans == 0, 0], prueba[r_kmeans == 0, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(prueba[r_kmeans == 3, 0], prueba[r_kmeans == 3, 1], c='black', marker='+', label = 'Iris-versicolour')
ax1.scatter(prueba[r_kmeans == 2, 0], prueba[r_kmeans == 2, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(prueba[r_kmeans == 1, 0], prueba[r_kmeans == 1, 1], c='green', marker='*', label = 'Iris-virginica')
ax1.scatter(prueba[r_kmeans == 4, 0], prueba[r_kmeans == 4, 1], c='blue', marker='o', label = 'Iris-virginica')

# Visualización de los centros de cada grupo
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("Prueba 5 Grupos")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in r_kmeans:
    if i == 0 : 
        salida.append(1)
    elif i == 3 or i == 2:
        salida.append(2)
    else:
        salida.append(3)

print(salida)

print(error(salida, etiquetas_prueba))


# %% [markdown]
# # Para 6 clusters

# %%
#Datos de entrenamiento normalizados con los dos descriptores seleccionados
x = df_TN.iloc[:, [2, 3]].values

centros = np.array([[0.1, 0.1],[0.2, 0.2], [0.3, 0.4], [0.5, 0.5], [0.7, 0.7], [0.8, 0.9]]) #Matriz que contiene las coordenadas de los centros

kmeans = KMeans(n_clusters = 6, init = centros, max_iter = 300, n_init = 10, random_state = 0)    #Entrenamiento para hallar los centros
y_kmeans = kmeans.fit_predict(x)             #Obtener la clasificación de los datos de entrenamiento
print(y_kmeans)                              # Se muestra la clasificación resultante de los datos de entrenamiento


fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización delos grupos
ax1.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], c='black', marker='+', label = 'Iris-versicolour')
ax1.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], c='green', marker='*', label = 'Iris-virginica')
ax1.scatter(x[y_kmeans == 5, 0], x[y_kmeans == 5, 1], c='blue', marker='o', label = 'Iris-virginica')
ax1.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], c='black', marker='x', label = 'Iris-virginica')

# Visualización de los centros de cada grupo
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("6 Grupos")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in y_kmeans:
    if i == 0: 
        salida.append(1)
    elif i == 2 or i == 3:
        salida.append(2)
    else:
        salida.append(3)

print(salida)
print("Error de entrenamiento",error(salida, etiquetas))

# %%
###############     Prueba   ################

#Datos de prueba normalizadoscon los dos descriptores seleccionados
prueba = df_PN.iloc[:, [2, 3]].values

r_kmeans = kmeans.predict(prueba)      #Clasificación de los datos de prueba en base a los centros obtenidos
print(r_kmeans)                        #Se muestra el resultado de la clasificación

fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización delos grupos
ax1.scatter(prueba[r_kmeans == 0, 0], prueba[r_kmeans == 0, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(prueba[r_kmeans == 2, 0], prueba[r_kmeans == 2, 1], c='black', marker='+', label = 'Iris-versicolour')
ax1.scatter(prueba[r_kmeans == 3, 0], prueba[r_kmeans == 3, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(prueba[r_kmeans == 4, 0], prueba[r_kmeans == 4, 1], c='green', marker='*', label = 'Iris-virginica')
ax1.scatter(prueba[r_kmeans == 5, 0], prueba[r_kmeans == 5, 1], c='blue', marker='o', label = 'Iris-virginica')
ax1.scatter(prueba[r_kmeans == 1, 0], prueba[r_kmeans == 1, 1], c='black', marker='x', label = 'Iris-virginica')

# Visualización de los centros de cada grupo
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("Prueba 6 Grupos")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in r_kmeans:
    if i == 0: 
        salida.append(1)
    elif i == 2 or i == 3:
        salida.append(2)
    else:
        salida.append(3)

print(salida)

print(error(salida, etiquetas_prueba))




# %% [markdown]
# # Punto 2

# %%
from fcmeans import FCM

# %% [markdown]
# # Para 3 clusters

# %%
#Datos de entrenamiento normalizados con los dos descriptores seleccionados
x = df_TN.iloc[:, [ 2, 3]].values


fcmean = FCM(n_clusters=3,m = 2.0, n_init=10, random_state=0)  #Parámetros para el algoritmo
y_fcmeans = fcmean.fit(x)               #Entrenamiento para hallar los centros
r_fcmeans = fcmean.predict(x)           #Obtener la clasificación de los datos de entrenamiento

print(r_fcmeans)                        # Se muestra la clasificación resultante de los datos de entrenamiento

fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización delos grupos
ax1.scatter(x[r_fcmeans == 0, 0], x[r_fcmeans == 0, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(x[r_fcmeans == 2, 0], x[r_fcmeans == 2, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(x[r_fcmeans == 1, 0], x[r_fcmeans == 1, 1], c='green', marker='*', label = 'Iris-virginica')

# Visualización de los centros de cada grupo
ax1.scatter(fcmean.centers[:, 0], fcmean.centers[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("3 Grupos FCM")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in r_fcmeans:
    if i == 0:
        salida.append(1)
    elif i == 2:
        salida.append(2)
    else:
        salida.append(3)

print(salida)

print("Error de entrenamiento", error(salida, etiquetas))

pert_fcmeans = fcmean.soft_predict(x)
print(np.round(pert_fcmeans * 100), 2)


# %%
###############     Prueba   ################

#Datos de prueba normalizadoscon los dos descriptores seleccionados
prueba = df_PN.iloc[:, [2, 3]].values

r_fcmeans = fcmean.predict(prueba)  #Clasificación de los datos de prueba en base a los centros obtenidos
print(r_fcmeans)                    #Se muestra el resultado de la clasificación

fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización delos grupos
ax1.scatter(prueba[r_fcmeans == 0, 0], prueba[r_fcmeans == 0, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(prueba[r_fcmeans == 2, 0], prueba[r_fcmeans == 2, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(prueba[r_fcmeans == 1, 0], prueba[r_fcmeans == 1, 1], c='green', marker='*', label = 'Iris-virginica')

# Visualización de los centros de cada grupo
ax1.scatter(fcmean.centers[:, 0], fcmean.centers[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("Prueba 3 Grupos FCM")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in r_fcmeans:
    if i == 0:
        salida.append(1)
    elif i == 2:
        salida.append(2)
    else:
        salida.append(3)

print(salida)

print(error(salida, etiquetas_prueba))

# Matriz de grados de pertenecia
pert_fcmeans = fcmean.soft_predict(prueba)
print(np.round(pert_fcmeans * 100), 2)

# %% [markdown]
# # Para 4 Clusters

# %%
#Datos de entrenamiento normalizados con los dos descriptores seleccionados
x = df_TN.iloc[:, [2, 3]].values

fcmean = FCM(n_clusters=4,m = 2.0, n_init=10, random_state=0)      #Parámetros para el algoritmo 
y_fcmeans = fcmean.fit(x)               #Entrenamiento para hallar los centros
r_fcmeans = fcmean.predict(x)           #Obtener la clasificación de los datos de entrenamiento

print(r_fcmeans)                # Se muestra la clasificación resultante de los datos de entrenamiento

fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización delos grupos
ax1.scatter(x[r_fcmeans == 2, 0], x[r_fcmeans == 2, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(x[r_fcmeans == 1, 0], x[r_fcmeans == 1, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(x[r_fcmeans == 3, 0], x[r_fcmeans == 3, 1], c='black', marker='*', label = 'Iris-versicolour')
ax1.scatter(x[r_fcmeans == 0, 0], x[r_fcmeans == 0, 1], c='green', marker='*', label = 'Iris-virginica')

# Visualización de los centros de cada grupo
ax1.scatter(fcmean.centers[:, 0], fcmean.centers[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("4 Grupos FCM")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in r_fcmeans:
    if i == 2:
        salida.append(1)
    elif i == 1 or i == 3:
        salida.append(2)
    else:
        salida.append(3)

print(salida)
print("Error de entrenamiento", error(salida, etiquetas))

# Matriz de grados de pertenecia
pert_fcmeans = fcmean.soft_predict(x)
print(np.round(pert_fcmeans * 100), 2)


# %%
###############     Prueba   ################

#Datos de prueba normalizadoscon los dos descriptores seleccionados
prueba = df_PN.iloc[:, [2, 3]].values

r_fcmeans = fcmean.predict(prueba)      #Clasificación de los datos de prueba en base a los centros obtenidos
print(r_fcmeans)                        #Se muestra el resultado de la clasificación

fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización delos grupos
ax1.scatter(prueba[r_fcmeans == 2, 0], prueba[r_fcmeans == 2, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(prueba[r_fcmeans == 1, 0], prueba[r_fcmeans == 1, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(prueba[r_fcmeans == 3, 0], prueba[r_fcmeans == 3, 1], c='black', marker='*', label = 'Iris-versicolour')
ax1.scatter(prueba[r_fcmeans == 0, 0], prueba[r_fcmeans == 0, 1], c='green', marker='*', label = 'Iris-virginica')

# Visualización de los centros de cada grupo
ax1.scatter(fcmean.centers[:, 0], fcmean.centers[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("Prueba 4 Grupos FCM")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in r_fcmeans:
    if i == 2:
        salida.append(1)
    elif i == 1 or i == 3:
        salida.append(2)
    else:
        salida.append(3)

print(salida)

print(error(salida, etiquetas_prueba))

# Matriz de grados de pertenecia
pert_fcmeans = fcmean.soft_predict(prueba)
print(np.round(pert_fcmeans * 100), 2)

# %% [markdown]
# # Para 5 clusters

# %%
#Datos de entrenamiento normalizados con los dos descriptores seleccionados
x = df_TN.iloc[:, [ 2, 3]].values

fcmean = FCM(n_clusters=5, m = 2.0, n_init=10, random_state=0)      #Parámetros para el algoritmo 
y_fcmeans = fcmean.fit(x)               #Entrenamiento para hallar los centros
r_fcmeans = fcmean.predict(x)           #Obtener la clasificación de los datos de entrenamiento

print(r_fcmeans)                        # Se muestra la clasificación resultante de los datos de entrenamiento

fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización delos grupos
ax1.scatter(x[r_fcmeans == 3, 0], x[r_fcmeans == 3, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(x[r_fcmeans == 2, 0], x[r_fcmeans == 2, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(x[r_fcmeans == 4, 0], x[r_fcmeans == 4, 1], c='green', marker='*', label = 'Iris-versicolour')
ax1.scatter(x[r_fcmeans == 1, 0], x[r_fcmeans == 1, 1], c='blue', marker='o', label = 'Iris-virginica')
ax1.scatter(x[r_fcmeans == 0, 0], x[r_fcmeans == 0, 1], c='black', marker='+', label = 'Iris-virginica')

# Visualización de los centros de cada grupo
ax1.scatter(fcmean.centers[:, 0], fcmean.centers[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("5 Grupos FCM")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in r_fcmeans:
    if  i == 3: 
        salida.append(1)
    elif i == 2 or i == 4:
        salida.append(2)
    else:
        salida.append(3)

print(salida)
print("Error de entrenamiento", error(salida, etiquetas))

# Matriz de grados de pertenecia
pert_fcmeans = fcmean.soft_predict(x)
print(np.round(pert_fcmeans * 100), 2)

# %%
###############     Prueba   ################

#Datos de prueba normalizadoscon los dos descriptores seleccionados
prueba = df_PN.iloc[:, [2, 3]].values

r_fcmeans = fcmean.predict(prueba)          #Clasificación de los datos de prueba en base a los centros obtenidos
print(r_fcmeans)                            #Se muestra el resultado de la clasificación

fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización delos grupos
ax1.scatter(prueba[r_fcmeans == 3, 0], prueba[r_fcmeans == 3, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(prueba[r_fcmeans == 2, 0], prueba[r_fcmeans == 2, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(prueba[r_fcmeans == 4, 0], prueba[r_fcmeans == 4, 1], c='green', marker='*', label = 'Iris-versicolour')
ax1.scatter(prueba[r_fcmeans == 1, 0], prueba[r_fcmeans == 1, 1], c='blue', marker='o', label = 'Iris-virginica')
ax1.scatter(prueba[r_fcmeans == 0, 0], prueba[r_fcmeans == 0, 1], c='black', marker='+', label = 'Iris-virginica')

# Visualización de los centros de cada grupo
ax1.scatter(fcmean.centers[:, 0], fcmean.centers[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("Prueba 5 Grupos FCM")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in r_fcmeans:
    if  i == 3: 
        salida.append(1)
    elif i == 2 or i == 4:
        salida.append(2)
    else:
        salida.append(3)

print(salida)

print(error(salida, etiquetas_prueba))

# Matriz de grados de pertenecia
pert_fcmeans = fcmean.soft_predict(prueba)
print(np.round(pert_fcmeans * 100), 2)

# %% [markdown]
# # Para 6 clusters

# %%
#Datos de entrenamiento normalizados con los dos descriptores seleccionados
x = df_TN.iloc[:, [ 2, 3]].values

fcmean = FCM(n_clusters=6, m = 2.0, n_init=10, random_state=0)      #Parámetros para el algoritmo 
y_fcmeans = fcmean.fit(x)                                           #Entrenamiento para hallar los centros
r_fcmeans = fcmean.predict(x)                                       #Obtener la clasificación de los datos de entrenamiento

print(r_fcmeans)                                    # Se muestra la clasificación resultante de los datos de entrenamiento

fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización de los grupos
ax1.scatter(x[r_fcmeans == 5, 0], x[r_fcmeans == 5, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(x[r_fcmeans == 0, 0], x[r_fcmeans == 0, 1], c='gray', marker='+', label = 'Iris-setosa')
ax1.scatter(x[r_fcmeans == 1, 0], x[r_fcmeans == 1, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(x[r_fcmeans == 3, 0], x[r_fcmeans == 3, 1], c='green', marker='*', label = 'Iris-versicolour')
ax1.scatter(x[r_fcmeans == 4, 0], x[r_fcmeans == 4, 1], c='blue', marker='o', label = 'Iris-virginica')
ax1.scatter(x[r_fcmeans == 2, 0], x[r_fcmeans == 2, 1], c='black', marker='x', label = 'Iris-virginica')


# Visualización de los centros de cada grupo
ax1.scatter(fcmean.centers[:, 0], fcmean.centers[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("6 Grupos FCM")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in r_fcmeans:
    if i == 5 or i == 0: 
        salida.append(1)
    elif i == 1 or i == 3:
        salida.append(2)
    else:
        salida.append(3)

print(salida)
print("Error de entrenamiento", error(salida, etiquetas))

# Matriz de grados de pertenecia
pert_fcmeans = fcmean.soft_predict(x)
print(np.round(pert_fcmeans * 100), 2)

# %%
###############     Prueba   ################

#Datos de prueba normalizadoscon los dos descriptores seleccionados
prueba = df_PN.iloc[:, [2, 3]].values

r_fcmeans = fcmean.predict(prueba)          #Clasificación de los datos de prueba en base a los centros obtenidos
print(r_fcmeans)                            #Se muestra el resultado de la clasificación

fig = plt.figure()
ax1 = fig.add_subplot(111)
# Visualización de los grupos
ax1.scatter(prueba[r_fcmeans == 5, 0], prueba[r_fcmeans == 5, 1], c='purple', marker='o', label = 'Iris-setosa')
ax1.scatter(prueba[r_fcmeans == 0, 0], prueba[r_fcmeans == 0, 1], c='gray', marker='+', label = 'Iris-setosa')
ax1.scatter(prueba[r_fcmeans == 1, 0], prueba[r_fcmeans == 1, 1], c='orange', marker='x', label = 'Iris-versicolour')
ax1.scatter(prueba[r_fcmeans == 3, 0], prueba[r_fcmeans == 3, 1], c='green', marker='*', label = 'Iris-versicolour')
ax1.scatter(prueba[r_fcmeans == 4, 0], prueba[r_fcmeans == 4, 1], c='blue', marker='o', label = 'Iris-virginica')
ax1.scatter(prueba[r_fcmeans == 2, 0], prueba[r_fcmeans == 2, 1], c='black', marker='x', label = 'Iris-virginica')

# Visualización de los centros de cada grupo
ax1.scatter(fcmean.centers[:, 0], fcmean.centers[:, 1], s = 100, c = 'red', label = 'Centros')
ax1.set_xlabel('X2')
ax1.set_ylabel('X3')
ax1.set_title("Prueba 6 Grupos FCM")
ax1.legend()

plt.show()

#Interpretación de la clasificación (El algoritmo le asigna en orden diferente)
salida = []
for i in r_fcmeans:
    if i == 5 or i == 0: 
        salida.append(1)
    elif i == 1 or i == 3:
        salida.append(2)
    else:
        salida.append(3)

print(salida)

print(error(salida, etiquetas_prueba))

# Matriz de grados de pertenecia
pert_fcmeans = fcmean.soft_predict(prueba)
print(np.round(pert_fcmeans * 100), 2)


