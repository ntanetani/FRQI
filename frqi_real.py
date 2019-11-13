from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, QuantumRegister
from qiskit.qasm import pi
from qiskit.tools.visualization import plot_histogram, circuit_drawer
import numpy as np

token = "e44ac171fe282a3c0d690386deb91d57028dcd020f13563bb33e0fa861bc3278c0b5ac6df23a4588aa723676ce76af1646f9e0118b2a497c5965fd43d8669336"
url = "https://q-console-api.mybluemix.net/api/Hubs/ibm-q-keio/Groups/keio-internal/Projects/keio-students"

IBMQ.enable_account(token, url)
# syntax and names are different from these two
# at the moment, only IBMQ shows real machine
backends = IBMQ.backends()
# backends = IBMQ.backends(simulator=False)
print("IBMQ backends:",backends)
# backends = Aer.backends()
# print("Aer backends:",backends)

q = QuantumRegister(3)
c = ClassicalRegister(3)
qc = QuantumCircuit(q,c)


def rry(circ, con1, con2, angle, t, c0, c1):

    circ.h(t)

    circ.rz(angle/2, t)

    if(con1 == 1):
        circ.x(c0)

    circ.cx(c0, t)

    if(con1 == 1):
        circ.x(c0)

    circ.rz(-angle/2, t)

    if(con2 == 1):
        circ.x(c1)

    circ.cx(c1, t)

    if(con2 == 1):
        circ.x(c1)

    circ.rz(angle/2, t)

    if(con1 == 1):
        circ.x(c0)

    circ.cx(c0, t)

    if(con1 == 1):
        circ.x(c0)

    circ.rz(-angle/2, t)

    circ.h(t)

for i in range(1,len(q)):
    qc.h(q[i])

rry(qc, 0, 0, np.pi/2, q[0], q[1], q[2])
rry(qc, 0, 1, np.pi/4, q[0], q[1], q[2])
rry(qc, 1, 0, np.pi/6, q[0], q[1], q[2])
rry(qc, 1, 1, np.pi/8, q[0], q[1], q[2])

for i in range(len(q)):
    qc.measure(q[i],c[i])

# backends = ['ibmq_20_tokyo', 'ibmq_qasm_simulator']

# Use this for the real machine
backend_sim = IBMQ.get_backend('ibmq_singapore')
# Use this for the simulation
#backend_sim = IBMQ.get_backend('ibmq_qasm_simulator')


result = execute(qc, backend_sim, shots=4096).result()
#circuit_drawer(qc).show()
plot_histogram(result.get_counts(qc))

print(result.get_counts(qc))
