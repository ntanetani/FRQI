from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, QuantumRegister
from qiskit.qasm import pi
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit import execute, Aer, BasicAer
import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mutual_info_score, r2_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_num = 0
plt.imshow(x_train[img_num], cmap='gray')
#plt.savefig('mnistimg'+str(img_num)+'.png')
#plt.show()

# 2次元データを数値に変換
x_train = x_train.reshape(60000, 784)
# 型変換
x_train = x_train.astype('float64')
#正規化(0~1)
x_train /= 255.0
x_train = np.arcsin(x_train)
backends = Aer.backends()
#print("Aer backends:",backends)

qubit = 12
qc = QuantumCircuit(qubit,qubit)

def margolus(circ, t, c0, c1):
        circ.ry(np.pi/4,t)
        circ.cx(c0, t)
        circ.ry(np.pi/4,t)
        circ.cx(c1, t)
        circ.ry(-np.pi/4,t)
        circ.cx(c0, t)
        circ.ry(-np.pi/4,t)

def ccry(circ, angle, t, c0, c1):
        circ.cu3(angle/2, 0, 0, c1, t)
        circ.cx(c1, c0)
        circ.cu3(-angle/2, 0, 0, c0, t)
        circ.cx(c1, c0)
        circ.cu3(angle/2, 0, 0, c0, t)

def mary(circ, angle, t, c0, c1):
        circ.ry(angle/4,t)
        circ.cx(c0, t)
        circ.ry(-angle/4,t)
        circ.cx(c1, t)
        circ.ry(angle/4,t)
        circ.cx(c0, t)
        circ.ry(-angle/4,t)
        circ.cx(c1, t)

def cccry(circ, angle, t, a, c0, c1, c2):
        margolus(circ, a, c1, c2)
        mary(circ, angle, t, a, c0)
        margolus(circ, a, c1, c2)

def mary_4(circ, angle, t, c0, c1, c2):
        circ.h(t)
        circ.t(t)
        circ.cx(c0,t)
        circ.tdg(t)
        circ.h(t)
        circ.cx(c1,t)
        circ.rz(angle/4,t)
        circ.cx(c2,t)
        circ.rz(-angle/4,t)
        circ.cx(c1,t)
        circ.rz(angle/4,t)
        circ.cx(c2,t)
        circ.rz(-angle/4,t)
        circ.h(t)
        circ.t(t)
        circ.cx(c0,t)
        circ.tdg(t)
        circ.h(t)

def c10ry(circ, angle, controls, q):

        clist = []

        for i in controls:
                clist.append(int(i))

        for i in range(len(clist)):
                if clist[i] == 0:
                        circ.x(q[len(clist)-i+1])

        margolus(circ, q[1], q[2], q[3])
        circ.x(q[2])
        circ.x(q[3])
        margolus(circ, q[3], q[4], q[5])
        circ.x(q[4])
        circ.x(q[5])
        margolus(circ, q[5], q[6], q[7])
        circ.x(q[6])
        circ.x(q[7])
        
        margolus(circ, q[7], q[10], q[11])
        margolus(circ, q[6], q[8], q[9])
        margolus(circ, q[4], q[6], q[7])
        margolus(circ, q[2], q[4], q[5])

        mary_4(circ, angle, q[0], q[1], q[2], q[3])

        margolus(circ, q[2], q[4], q[5])
        margolus(circ, q[4], q[6], q[7])
        margolus(circ, q[6], q[8], q[9])
        margolus(circ, q[7], q[10], q[11])
        
        circ.x(q[7])
        circ.x(q[6])
        margolus(circ, q[5], q[6], q[7])
        circ.x(q[5])
        circ.x(q[4])
        margolus(circ, q[3], q[4], q[5])
        circ.x(q[3])
        circ.x(q[2])
        margolus(circ, q[1], q[2], q[3])

        for i in range(len(clist)):
                if clist[i] == 0:
                        circ.x(q[len(clist)-i+1])


qc.h(range(2,qubit))

for i in range(len(x_train[img_num])):
        if x_train[img_num][i] != 0:
                c10ry(qc, 2 * x_train[img_num][i], format(i, '010b'), list(range(12)))

qc.measure(range(qubit),range(qubit))

backend_sim = Aer.get_backend('qasm_simulator')
#print(qc.depth())
numOfShots = 1024000
result = execute(qc, backend_sim, shots=numOfShots).result()
#circuit_drawer(qc).show()
#plot_histogram(result.get_counts(qc))

print(result.get_counts(qc))

genimg = np.array([])
for i in range(len(x_train[img_num])):
        try:
                genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'01']/numOfShots)])
        except KeyError:
                genimg = np.append(genimg,[0.0])
#print(genimg)
#print(genimg.shape)
#正規化(戻し)
genimg *= 32.0 * 255.0
x_train = np.sin(x_train)
x_train *= 255.0
# 型変換
genimg = genimg.astype('int')

print(mutual_info_score(x_train[img_num], genimg))
gosa = genimg - x_train[img_num]
for i in range(len(x_train[img_num])):
        if x_train[img_num][i] != 0:
                gosa[i] = np.abs(gosa[i] / x_train[img_num][i])
        else:
                gosa[i] = 0

# 28*28
genimg = genimg.reshape((28,28))
plt.imshow(genimg, cmap='gray', vmin=0, vmax=255)
plt.savefig('gen_'+str(img_num)+'.png')
plt.show()
gosa = gosa.reshape((28,28))
plt.imshow(gosa, cmap='jet', vmin=0, vmax=1)
plt.colorbar()
plt.savefig('gosa_'+str(img_num)+'.png')
plt.show()