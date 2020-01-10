import numpy as np
import matplotlib.pyplot as plt
import nengo
from nengo.dists import Choice
from nengo.processes import Piecewise


def AND_OPERATION():
    with nengo.Network() as AND:
        AND.I = nengo.Ensemble(800, dimensions=2, radius=1.6)
        AND.O = nengo.Ensemble(400, dimensions=1, radius=1.1)
        AND.PROD = nengo.Ensemble(800, dimensions=1, radius=1)
        AND.MIN = nengo.Ensemble(800, dimensions=1, radius=1)

        # Fuzzy set AND operations. You can try different AND operations replacing the bdif_func at the connections
        def prod_func(x):
            x_arr = np.array(x)
            return np.prod(x_arr)

        def min_func(x):
            x_arr = np.array(x)
            return np.min(x_arr)

        def bdif_func(x):
            x_arr = np.array(x)
            return np.maximum(0, x[0]+x[1]-1)

        nengo.Connection(AND.I, AND.O, function=prod_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return AND

model = nengo.Network(label='Fuzzy operations')
with model:  
    time = np.arange(0, 2.5, 0.01)
    input_1 = time - 1.5
    input_2 = -time/2 + 0.75
    input_3 = (time - 1.5)**2

    stim_1 = nengo.Node(Piecewise(dict(zip(time, input_1))))
    stim_2 = nengo.Node(Piecewise(dict(zip(time, input_2))))
    stim_3 = nengo.Node(Piecewise(dict(zip(time, input_3))))

    prod_op = AND_OPERATION()
    nengo.Connection(stim_1, prod_op.I[0])
    nengo.Connection(stim_2, prod_op.I[1])

    min_op = AND_OPERATION()
    nengo.Connection(stim_1, min_op.I[0])
    nengo.Connection(stim_2, min_op.I[1])

    bdif_op = AND_OPERATION()
    nengo.Connection(stim_1, bdif_op.I[0])
    nengo.Connection(stim_2, bdif_op.I[1])

with model:
    pr_input_1 = nengo.Probe(stim_1, synapse=0.01)
    pr_input_2 = nengo.Probe(stim_2, synapse=0.01)
    pr_input_3 = nengo.Probe(stim_3, synapse=0.01)

    pr_prod_output = nengo.Probe(prod_op.O, synapse=0.01)  
    pr_min_output = nengo.Probe(min_op.O, synapse=0.01)  
    pr_bdif_output = nengo.Probe(bdif_op.O, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(2.5)

t = sim.trange()
plt.figure()
plt.subplot(411)
plt.plot(t[500:]-1.5, sim.data[pr_input_1][500:], c='r', label="Input 1")
plt.plot(t[500:]-1.5, sim.data[pr_input_2][500:], c='g', label="Input 2")
plt.plot(t[500:]-1.5, sim.data[pr_input_3][500:], c='b', label="Input 3")
plt.xlim(right=1)
plt.xlabel("Time (s)")
plt.ylabel("Inputs")
plt.legend(loc="best")
plt.grid(True)

plt.subplot(412)
plt.plot(t[500:]-1.5, sim.data[pr_prod_output][500:], color = '#888800', label="prod operation")
plt.xlim(right=1)
plt.xlabel("Time (s)")
plt.ylabel("prod()")
plt.legend(loc="best")
plt.grid(True)

plt.subplot(413)
plt.plot(t[500:]-1.5, sim.data[pr_min_output][500:], color = '#880088', label="min operation")
plt.xlim(right=1)
plt.xlabel("Time (s)")
plt.ylabel("min()")
plt.legend(loc="best")
plt.grid(True)

plt.subplot(414)
plt.plot(t[500:]-1.5, sim.data[pr_bdif_output][500:], color = '#008888', label="bdif operation")
plt.xlim(right=1)
plt.xlabel("Time (s)")
plt.ylabel("bdif()")
plt.legend(loc="best")
plt.grid(True)
plt.show()
