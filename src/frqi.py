from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, transpile
from qiskit.qasm import pi
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit import execute, Aer, BasicAer
from qiskit.aqua.circuits.gates.relative_phase_toffoli import rccx, rcccx
import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mutual_info_score, r2_score

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

def mary_4(self, angle, t, c0, c1, c2):
        self.h(t)
        self.t(t)
        self.cx(c0,t)
        self.tdg(t)
        self.h(t)
        self.cx(c1,t)
        self.rz(angle/4,t)
        self.cx(c2,t)
        self.rz(-angle/4,t)
        self.cx(c1,t)
        self.rz(angle/4,t)
        self.cx(c2,t)
        self.rz(-angle/4,t)
        self.h(t)
        self.t(t)
        self.cx(c0,t)
        self.tdg(t)
        self.h(t)

def mary_8(self, angle, t, c0, c1, c2, c3, c4, c5, c6):
        self.h(t)
        self.t(t)
        self.rccx(t, c0, c1)
        self.tdg(t)
        self.h(t)
        self.rccx(t, c2, c3)
        self.rz(angle/4,t)
        self.rcccx(t, c4, c5, c6)
        self.rz(-angle/4,t)
        self.rccx(t, c2, c3)
        self.rz(angle/4,t)
        self.rcccx(t, c4, c5, c6)
        self.rz(-angle/4,t)
        self.h(t)
        self.t(t)
        self.rccx(t, c0, c1)
        self.tdg(t)
        self.h(t)

def c10ry(circ, angle, bin, target, anc, controls):

        clist = []

        for i in bin:
                clist.append(int(i))

        clist = list(reversed(clist))

        for i in range(len(clist)):
                if clist[i] == 0:
                        circ.x(controls[-i-1])

        rccx(circ, anc, controls[0], controls[1])
        circ.x(controls[0])
        circ.x(controls[1])
        rccx(circ, controls[1], controls[2], controls[3])
        circ.x(controls[2])
        circ.x(controls[3])
        rccx(circ, controls[3], controls[4], controls[5])
        circ.x(controls[4])
        circ.x(controls[5])
        
        rccx(circ, controls[5], controls[8], controls[9])
        rccx(circ, controls[4], controls[6], controls[7])
        rccx(circ, controls[2], controls[4], controls[5])
        rccx(circ, controls[0], controls[2], controls[3])

        mary_4(circ, angle, target, anc, controls[0], controls[1])

        rccx(circ, controls[0], controls[2], controls[3])
        rccx(circ, controls[2], controls[4], controls[5])
        rccx(circ, controls[4], controls[6], controls[7])
        rccx(circ, controls[5], controls[8], controls[9])
        
        circ.x(controls[5])
        circ.x(controls[4])
        rccx(circ, controls[3], controls[4], controls[5])
        circ.x(controls[3])
        circ.x(controls[2])
        rccx(circ, controls[1], controls[2], controls[3])
        circ.x(controls[1])
        circ.x(controls[0])
        rccx(circ, anc, controls[0], controls[1])

        for i in range(len(clist)):
                if clist[i] == 0:
                        circ.x(controls[-i-1])


def c10mary(circ, angle, bin, target, anc, controls):
        clist = []

        for i in bin:
                clist.append(int(i))

        for i in range(len(clist)):
                if clist[i] == 0:
                        circ.x(controls[-i-1])

        rccx(circ, anc, controls[4], controls[5])
        circ.x(controls[4])
        circ.x(controls[5])
        rccx(circ, controls[4], controls[6], controls[7])
        rccx(circ, controls[5], controls[8], controls[9])


        mary_8(circ, angle, target, anc, controls[0], controls[1], controls[2], controls[3], controls[4], controls[5])

        rccx(circ, controls[5], controls[8], controls[9])
        rccx(circ, controls[4], controls[6], controls[7])
        circ.x(controls[5])
        circ.x(controls[4])
        rccx(circ, anc, controls[4], controls[5])

        for i in range(len(clist)):
                if clist[i] == 0:
                        circ.x(controls[-i-1])
def mcry(circ, angle, bin, target, controls, anc):

        assert len(bin) == len(controls), "error"
        assert len(bin) > 5, "ERROR"

        clist = []

        for i in bin:
                clist.append(int(i))

        for i in range(len(clist)):
                if clist[i] == 0:
                        circ.x(controls[-i-1])
        
        for i in range(0, len(clist)-4+len(clist)%2, 2):
                if i == 0:
                        circ.ccx(controls[i], controls[i+1], anc[0])
                else:
                        circ.ccx(controls[i], controls[i+1], controls[i-1])

                circ.x(controls[i])
                circ.x(controls[i+1])
        
        if (len(clist)%2) == 0:
                circ.ccx(controls[-1], controls[-2], controls[-5])
        else:
                circ.cx(controls[-1], controls[-4])
        
        for i in range(6-len(clist)%2, len(clist)+1, 2):
                circ.ccx(controls[-i+3], controls[-i+2], controls[-i])

        mary_4(circ, angle, target, anc[0], controls[0], controls[1])

        for i in range(6-len(clist)%2, len(clist)+1, 2)[::-1]:
                circ.ccx(controls[-i+3], controls[-i+2], controls[-i])

        if (len(clist)%2) == 0:
                circ.ccx(controls[-1], controls[-2], controls[-5])
        else:
                circ.cx(controls[-1], controls[-4])

        for i in range(0, len(clist)-4+len(clist)%2, 2)[::-1]:

                circ.x(controls[i])
                circ.x(controls[i+1])

                if i == 0:
                        circ.ccx(controls[i], controls[i+1], anc[0])
                else:
                        circ.ccx(controls[i], controls[i+1], controls[i-1])

        for i in range(len(clist)):
                if clist[i] == 0:
                        circ.x(controls[-i-1])

def rmcry(self, angle, bin, target, controls, anc):

        assert len(bin) == len(controls), "error"
        assert len(bin) > 5, "ERROR"

        clist = []

        for i in bin:
                clist.append(int(i))

        for i in range(len(clist)):
                if clist[i] == 0:
                        self.x(controls[-i-1])
        
        for i in range(0, len(clist)-4+len(clist)%2, 2):
                if i == 0:
                        self.ccx(controls[i], controls[i+1], anc[0])
                else:
                        self.ccx(controls[i], controls[i+1], controls[i-1])

                self.x(controls[i])
                self.x(controls[i+1])
        
        if (len(clist)%2) == 0:
                self.rccx(controls[-1], controls[-2], controls[-5])
        else:
                self.cx(controls[-1], controls[-4])
        
        for i in range(6-len(clist)%2, len(clist)+1, 2):
                self.rccx(controls[-i+3], controls[-i+2], controls[-i])

        self.mary_4(angle, target, anc[0], controls[0], controls[1])

        for i in reversed(range(6-len(clist)%2, len(clist)+1, 2)):
                self.rccx(controls[-i+3], controls[-i+2], controls[-i])

        if (len(clist)%2) == 0:
                self.rccx(controls[-1], controls[-2], controls[-5])
        else:
                self.cx(controls[-1], controls[-4])

        for i in reversed(range(0, len(clist)-4+len(clist)%2, 2)):
                
                self.x(controls[i])
                self.x(controls[i+1])

                if i == 0:
                        self.rccx(controls[i], controls[i+1], anc[0])
                else:
                        self.rccx(controls[i], controls[i+1], controls[i-1])

        for i in range(len(clist)):
                if clist[i] == 0:
                        self.x(controls[-i-1])

QuantumCircuit.mary_4 = mary_4
QuantumCircuit.rmcry = rmcry

if __name__ == '__main__':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_num = 0

        #show original image
        plt.imshow(x_train[img_num], cmap='gray')
        #plt.savefig('mnistimg'+str(img_num)+'.png')
        #plt.show()

        # 2-dimentional data convert to 1-dimentional array
        x_train = x_train.reshape(60000, 784)
        # change type
        x_train = x_train.astype('float64')
        # Normalization(0~pi/2)
        x_train /= 255.0
        x_train = np.arcsin(x_train)

        backends = Aer.backends()
        #print("Aer backends:",backends)

        qubit = 12
        q = QuantumRegister(qubit, "q")
        c = ClassicalRegister(qubit, "c")
        qc = QuantumCircuit(q,c)


        # apply hadamard gates
        qc.h(range(2,qubit))

        # apply c10Ry gates (representing color data)
        for i in range(len(x_train[img_num])):
                if x_train[img_num][i] != 0:
                        #c10ry(qc, 2 * x_train[img_num][i], format(i, '010b'), 0, 1, [i for i in range(2,12)])
                        qc.rmcry(2 * x_train[img_num][i], format(i, '010b'), q[0], q[2:12], [q[1]])

        qc.measure(q[0:qubit],c[0:qubit])

        backend_sim = Aer.get_backend('qasm_simulator')
        #print(qc.depth())
        numOfShots = 1024000
        result = execute(qc, backend_sim, shots=numOfShots).result()
        #circuit_drawer(qc).show()a
        #plot_histogram(result.get_counts(qc))

        print(result.get_counts(qc))

        # generated image
        genimg = np.array([])

        #### decode
        for i in range(len(x_train[img_num])):
                try:
                        genimg = np.append(genimg,[np.sqrt(result.get_counts(qc)[format(i, '010b')+'01']/numOfShots)])
                except KeyError:
                        genimg = np.append(genimg,[0.0])

        # inverse nomalization
        genimg *= 32.0 * 255.0
        x_train = np.sin(x_train)
        x_train *= 255.0

        # convert type
        genimg = genimg.astype('int')

        # back to 2-dimentional data
        genimg = genimg.reshape((28,28))

        plt.imshow(genimg, cmap='gray', vmin=0, vmax=255)
        plt.savefig('gen_'+str(img_num)+'.png')
        plt.show()