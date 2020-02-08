from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, QuantumRegister
from qiskit.qasm import pi
from qiskit.tools.visualization import plot_histogram, circuit_drawer
import numpy as np
import random

token = "e44ac171fe282a3c0d690386deb91d57028dcd020f13563bb33e0fa861bc3278c0b5ac6df23a4588aa723676ce76af1646f9e0118b2a497c5965fd43d8669336"
url = "https://api-qcon.quantum-computing.ibm.com/api/Hubs/ibm-q-keio/Groups/keio-internal/Projects/keio-students"

IBMQ.enable_account(token, url)
# syntax and names are different from these two
# at the moment, only IBMQ shows real machine
backends = IBMQ.backends()
# backends = IBMQ.backends(simulator=False)
print("IBMQ backends:",backends)
# backends = Aer.backends()
# print("Aer backends:",backends)

q = QuantumRegister(12)
c = ClassicalRegister(12)
qc = QuantumCircuit(q,c)


qc.h(q[0])
qc.h(q[1])
qc.h(q[2])
qc.h(q[3])
qc.h(q[4])
qc.h(q[5])
qc.h(q[6])
qc.h(q[7])
qc.h(q[8])
qc.h(q[9])
qc.h(q[10])
for i in range(50000):
    a = random.randrange(12)
    b = random.randrange(12)
    if a != b:
        qc.cx(q[a],q[b])


qc.measure(q[0],c[0])
qc.measure(q[1],c[1])
qc.measure(q[2],c[2])
qc.measure(q[3],c[3])
qc.measure(q[4],c[4])
qc.measure(q[5],c[5])
qc.measure(q[6],c[6])
qc.measure(q[7],c[7])
qc.measure(q[8],c[8])
qc.measure(q[9],c[9])
qc.measure(q[10],c[10])
qc.measure(q[11],c[11])


backend_sim = IBMQ.get_backend('ibmqx_hpc_qasm_simulator')

result = execute(qc, backend_sim, shots=8192).result()
#circuit_drawer(qc).show()
#plot_histogram(result.get_counts(qc))

print(result.get_counts(qc))
