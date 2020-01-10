import numpy as np
import matplotlib.pyplot as plt
import nengo
from nengo.dists import Choice
from nengo.processes import Piecewise


def OR_OPERATION():
    with nengo.Network() as OR:
        OR.I = nengo.Ensemble(800, dimensions=2, radius=1.6)
        OR.O = nengo.Ensemble(400, dimensions=1, radius=1.1)
        OR.asum = nengo.Ensemble(800, dimensions=1, radius=1)
        OR.max = nengo.Ensemble(800, dimensions=1, radius=1)

        # Fuzzy set OR operations. You can try different OR operations replacing the bsum_func at the connections
        def asum_func(x):
            return x[0] + x[1] - x[0]*x[1]

        def max_func(x):
            x_arr = np.array(x)
            return np.max(x_arr)

        def bsum_func(x):
            x_arr = np.array(x)
            return np.minimum(1, x[0]+x[1])

        nengo.Connection(OR.I, OR.O, function=asum_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return OR

model = nengo.Network(label='Fuzzy operations')
with model:  
    time = np.arange(0, 2.5, 0.01)
    input_1 = time - 1.5
    input_2 = -time/2 + 0.75
    input_3 = (time - 1.5)**2

    stim_1 = nengo.Node(Piecewise(dict(zip(time, input_1))))
    stim_2 = nengo.Node(Piecewise(dict(zip(time, input_2))))
    stim_3 = nengo.Node(Piecewise(dict(zip(time, input_3))))

    asum_op = OR_OPERATION()
    nengo.Connection(stim_1, asum_op.I[0])
    nengo.Connection(stim_2, asum_op.I[1])

    max_op = OR_OPERATION()
    nengo.Connection(stim_1, max_op.I[0])
    nengo.Connection(stim_2, max_op.I[1])

    bsum_op = OR_OPERATION()
    nengo.Connection(stim_1, bsum_op.I[0])
    nengo.Connection(stim_2, bsum_op.I[1])

with model:
    pr_input_1 = nengo.Probe(stim_1, synapse=0.01)
    pr_input_2 = nengo.Probe(stim_2, synapse=0.01)
    pr_input_3 = nengo.Probe(stim_3, synapse=0.01)

    pr_asum_output = nengo.Probe(asum_op.O, synapse=0.01)  
    pr_max_output = nengo.Probe(max_op.O, synapse=0.01)  
    pr_bsum_output = nengo.Probe(bsum_op.O, synapse=0.01)

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
plt.plot(t[500:]-1.5, sim.data[pr_asum_output][500:], color = '#888800', label="asum operation")
plt.xlim(right=1)
plt.xlabel("Time (s)")
plt.ylabel("asum()")
plt.legend(loc="best")
plt.grid(True)

plt.subplot(413)
plt.plot(t[500:]-1.5, sim.data[pr_max_output][500:], color = '#880088', label="max operation")
plt.xlim(right=1)
plt.xlabel("Time (s)")
plt.ylabel("max()")
plt.legend(loc="best")
plt.grid(True)

plt.subplot(414)
plt.plot(t[500:]-1.5, sim.data[pr_bsum_output][500:], color = '#008888', label="bsum operation")
plt.xlim(right=1)
plt.xlabel("Time (s)")
plt.ylabel("bsum()")
plt.legend(loc="best")
plt.grid(True)
plt.show()
