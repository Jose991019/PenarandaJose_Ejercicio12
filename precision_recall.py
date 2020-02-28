import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)


data = imagenes.reshape((n_imagenes, -1)) 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
y_train[y_train!=1] = 0
y_test[y_test!=1]=0

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

numero = 1
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

clf = LinearDiscriminantAnalysis()


proyeccion_train = np.dot(x_train,vectores)
proyeccion_test = np.dot(x_test,vectores)

    
clf.fit(proyeccion_train[:,:10], y_train.T)
probabilidades = clf.predict_proba(proyeccion_test[:,:10])
precision1, recall1, treshold1 = sklearn.metrics.precision_recall_curve(y_test, probabilidades[:,1])
f1_score1 = 2*precision1*recall1/(precision1+recall1)


cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

clf = LinearDiscriminantAnalysis()


proyeccion_train = np.dot(x_train,vectores)
proyeccion_test = np.dot(x_test,vectores)

    
clf.fit(proyeccion_train[:,:10], y_train.T)
probabilidades_todos = clf.predict_proba(proyeccion_test[:,:10])
precision_todos, recall_todos, treshold_todos = sklearn.metrics.precision_recall_curve(y_test, probabilidades_todos[:,1])
f1_score_todos = 2*precision_todos*recall_todos/(precision_todos+recall_todos)

    
    
numero = 0
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

clf = LinearDiscriminantAnalysis()


proyeccion_train = np.dot(x_train,vectores)
proyeccion_test = np.dot(x_test,vectores)

    
clf.fit(proyeccion_train[:,:10], y_train.T)
probabilidades = clf.predict_proba(proyeccion_test[:,:10])
precision0, recall0, treshold0 = sklearn.metrics.precision_recall_curve(y_test, probabilidades[:,1])
f1_score0 = 2*precision0*recall0/(precision0+recall0)
    


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.plot(treshold1,f1_score1[:-1], label = 'Solo 1')
indice = np.where(f1_score1[:-1] == np.max(f1_score1[:-1]))
print(indice)
plt.scatter(treshold1[indice], f1_score1[:-1][indice], color = 'r')
plt.legend()
plt.xlabel('Probabilidad')
plt.ylabel('F1')
plt.subplot(1,2,2)
plt.plot(recall1,precision1, label = 'solo1')
plt.legend()
plt.scatter(recall1[indice], precision1[indice], color = 'r')
plt.xlabel('Recall')
plt.ylabel('Precisión')


plt.subplot(1,2,1)
plt.plot(treshold_todos,f1_score_todos[:-1], label = 'Todos')
plt.legend()
indice = np.where(f1_score_todos[:-1] == np.max(f1_score_todos[:-1]))
print(indice)
plt.scatter(treshold_todos[indice], f1_score_todos[:-1][indice], color = 'r')
plt.xlabel('Probabilidad')
plt.ylabel('F1')
plt.subplot(1,2,2)
plt.plot(recall_todos,precision_todos, label = 'Todos')
plt.scatter(recall_todos[indice], precision_todos[indice], color = 'r')
plt.xlabel('Recall')
plt.ylabel('Precisión')
plt.legend()


plt.subplot(1,2,1)
plt.plot(treshold0,f1_score0[:-1], label = 'Solo 0')
plt.legend()

indice = np.where(f1_score0[:-1] == np.max(f1_score0[:-1]))
plt.scatter(treshold0[indice], f1_score0[:-1][indice], color = 'r')
print(indice)
plt.xlabel('Probabilidad')
plt.ylabel('F1')
plt.subplot(1,2,2)
plt.plot(recall0,precision0, label = 'Solo 0')
plt.scatter(recall0[indice], precision0[indice], color = 'r')
plt.xlabel('Recall')
plt.ylabel('Precisión')
plt.legend()



plt.savefig('F1_prec_recall.png')