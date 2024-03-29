from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import transpile
from qiskit.qasm import pi
from qiskit import execute, Aer, BasicAer
import numpy as np
import Gates

def margolus(circ, t, c0, c1):
        circ.ry(np.pi/4,t)
        circ.cx(c0, t)
        circ.ry(np.pi/4,t)
        circ.cx(c1, t)
        circ.ry(-np.pi/4,t)
        circ.cx(c0, t)
        circ.ry(-np.pi/4,t)

def rccx(circ, t, c0, c1):
        circ.h(t)
        circ.t(t)
        circ.cx(c0, t)
        circ.tdg(t)
        circ.cx(c1, t)
        circ.t(t)
        circ.cx(c0, t)
        circ.tdg(t)
        circ.h(t)

def rcccx(circ, t, c0, c1, c2):
        circ.h(t)
        circ.t(t)
        circ.cx(c0, t)
        circ.tdg(t)
        circ.h(t)
        circ.cx(c1, t)
        circ.t(t)
        circ.cx(c2, t)
        circ.tdg(t)
        circ.cx(c1, t)
        circ.t(t)
        circ.cx(c2, t)
        circ.tdg(t)
        circ.h(t)
        circ.t(t)
        circ.cx(c0, t)
        circ.tdg(t)
        circ.h(t)

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

def mary_8(circ, angle, t, c0, c1, c2, c3, c4, c5, c6):
        circ.h(t)
        circ.t(t)
        rccx(circ, t, c0, c1)
        circ.tdg(t)
        circ.h(t)
        rccx(circ, t, c2, c3)
        circ.rz(angle/4,t)
        rcccx(circ, t, c4, c5, c6)
        circ.rz(-angle/4,t)
        rccx(circ, t, c2, c3)
        circ.rz(angle/4,t)
        rcccx(circ, t, c4, c5, c6)
        circ.rz(-angle/4,t)
        circ.h(t)
        circ.t(t)
        rccx(circ, t, c0, c1)
        circ.tdg(t)
        circ.h(t)

def mary_11(circ, angle, bin, t, cs):

        clist = []

        for i in bin:
                clist.append(int(i))

        clist = list(reversed(clist))

        for i in range(len(clist)):
                if clist[i] == 0:
                        circ.x(cs[-i-1])

        circ.h(t)
        circ.t(t)

        circ.h(t)
        circ.t(t)
        circ.cx(cs[0], t)
        circ.tdg(t)
        circ.h(t)
        circ.cx(cs[1], t)
        circ.t(t)
        circ.cx(cs[2], t)
        circ.tdg(t)
        circ.cx(cs[1], t)
        circ.t(t)
        circ.cx(cs[2], t)
        circ.tdg(t)
        circ.h(t)
        circ.t(t)
        circ.cx(cs[0], t)
        circ.tdg(t)
        circ.h(t)

        circ.tdg(t)
        circ.h(t)

        circ.h(t)
        circ.t(t)
        circ.cx(cs[3], t)
        circ.tdg(t)
        circ.h(t)
        circ.cx(cs[4], t)
        circ.t(t)
        circ.cx(cs[5], t)
        circ.tdg(t)
        circ.cx(cs[4], t)
        circ.t(t)
        circ.cx(cs[5], t)
        circ.tdg(t)
        circ.h(t)
        circ.t(t)
        circ.cx(cs[3], t)
        circ.tdg(t)
        circ.h(t)

        circ.rz(angle/4,t)

        circ.h(t)
        circ.t(t)
        circ.cx(cs[6], t)
        circ.tdg(t)
        circ.h(t)
        circ.cx(cs[7], t)
        circ.t(t)

        circ.h(t)
        circ.t(t)
        circ.cx(cs[8], t)
        circ.tdg(t)
        circ.cx(cs[9], t)
        circ.t(t)
        circ.cx(cs[8], t)
        circ.tdg(t)
        circ.h(t)

        circ.tdg(t)
        circ.cx(cs[7], t)
        circ.t(t)

        circ.h(t)
        circ.t(t)
        circ.cx(cs[8], t)
        circ.tdg(t)
        circ.cx(cs[9], t)
        circ.t(t)
        circ.cx(cs[8], t)
        circ.tdg(t)
        circ.h(t)

        circ.tdg(t)
        circ.h(t)
        circ.t(t)
        circ.cx(cs[6], t)
        circ.tdg(t)
        circ.h(t)

        circ.rz(-angle/4,t)

        circ.h(t)
        circ.t(t)
        circ.cx(cs[3], t)
        circ.tdg(t)
        circ.h(t)
        circ.cx(cs[4], t)
        circ.t(t)
        circ.cx(cs[5], t)
        circ.tdg(t)
        circ.cx(cs[4], t)
        circ.t(t)
        circ.cx(cs[5], t)
        circ.tdg(t)
        circ.h(t)
        circ.t(t)
        circ.cx(cs[3], t)
        circ.tdg(t)
        circ.h(t)

        circ.rz(angle/4,t)

        circ.h(t)
        circ.t(t)
        circ.cx(cs[6], t)
        circ.tdg(t)
        circ.h(t)
        circ.cx(cs[7], t)
        circ.t(t)

        circ.h(t)
        circ.t(t)
        circ.cx(cs[8], t)
        circ.tdg(t)
        circ.cx(cs[9], t)
        circ.t(t)
        circ.cx(cs[8], t)
        circ.tdg(t)
        circ.h(t)

        circ.tdg(t)
        circ.cx(cs[7], t)
        circ.t(t)

        circ.h(t)
        circ.t(t)
        circ.cx(cs[8], t)
        circ.tdg(t)
        circ.cx(cs[9], t)
        circ.t(t)
        circ.cx(cs[8], t)
        circ.tdg(t)
        circ.h(t)

        circ.tdg(t)
        circ.h(t)
        circ.t(t)
        circ.cx(cs[6], t)
        circ.tdg(t)
        circ.h(t)

        circ.rz(-angle/4,t)
        circ.h(t)
        circ.t(t)

        circ.h(t)
        circ.t(t)
        circ.cx(cs[0], t)
        circ.tdg(t)
        circ.h(t)
        circ.cx(cs[1], t)
        circ.t(t)
        circ.cx(cs[2], t)
        circ.tdg(t)
        circ.cx(cs[1], t)
        circ.t(t)
        circ.cx(cs[2], t)
        circ.tdg(t)
        circ.h(t)
        circ.t(t)
        circ.cx(cs[0], t)
        circ.tdg(t)
        circ.h(t)

        circ.tdg(t)
        circ.h(t)

        for i in range(len(clist)):
                if clist[i] == 0:
                        circ.x(cs[-i-1])

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


if __name__ == '__main__':
        qubit = 23
        target = QuantumRegister(1, "target")
        controls = QuantumRegister(21, "controls")
        anc = QuantumRegister(2, "anc")
        c = ClassicalRegister(23)
        qc = QuantumCircuit(target, controls, anc, c)
        # apply hadamard gates
        qc.h(controls)
        #qc.h(range(2,qubit))

        for i in range(1600*1124):
            #c10ry(qc, np.pi * (i / 784), format(i, '010b'), 0, 1, [i for i in range(2,12)])
            #mary_11(qc, np.pi * (i / 1600*1124), format(i, '010b'), 0, [i for i in range(1,11)])
            qc.rmcry(np.pi * (i / 1600*1124), format(i, '021b'), controls, target, anc)
        
        transpiled_circ = transpile(qc, basis_gates=['cx', 'u3'], optimization_level=0)
        print('depth:' + transpiled_circ.depth())
        print(transpiled_circ.count_ops())
        transpiled_circ2 = transpile(qc, basis_gates=['cx', 'u3'], optimization_level=1)
        print('transpiled_depth:' + transpiled_circ2.depth())
        print(transpiled_circ2.count_ops())