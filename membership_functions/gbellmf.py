import numpy as np
import matplotlib.pyplot as plt
import nengo
from nengo.dists import Choice
from nengo.processes import Piecewise


def gbellmf(F):
    with nengo.Network() as GBELLMF:
        GBELLMF.I = nengo.Ensemble(2400, dimensions=1, radius=1)
        GBELLMF.O = nengo.Ensemble(800, dimensions=1, radius=1)
        #GBELLMF.I.encoders=Choice([[1], [-1]])

        def gbellmf_func(x):
            return 1 / (1 + np.power(np.abs((x - F[2]) / F[0]), 2*F[1]))

        nengo.Connection(GBELLMF.I, GBELLMF.O, function=gbellmf_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
        return GBELLMF


model = nengo.Network(label='TFLC')
with model:
    time = np.arange(0, 2.5, 0.01)
    input = time - 1.5

    stim_red = nengo.Node(Piecewise(dict(zip(time, input))))
    stim_green = nengo.Node(Piecewise(dict(zip(time, input))))
    stim_blue = nengo.Node(Piecewise(dict(zip(time, input))))

    red = gbellmf([0.25, 8, -0.6])
    nengo.Connection(stim_red, red.I)

    green = gbellmf([0.35, 2, -0.1])
    nengo.Connection(stim_green, green.I)

    blue = gbellmf([0.25, 3, 0.3])
    nengo.Connection(stim_blue, blue.I)

with model:
    pr_red = nengo.Probe(red.O, synapse=0.01)
    pr_green = nengo.Probe(green.O, synapse=0.01)
    pr_blue = nengo.Probe(blue.O, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(2.5)

t = sim.trange()
print(len(sim.data[pr_red]))
plt.figure()
plt.plot(t[500:]-1.5, sim.data[pr_red][500:], c='r', label="Red")
plt.plot(t[500:]-1.5, sim.data[pr_green][500:], c='g', label="Green")
plt.plot(t[500:]-1.5, sim.data[pr_blue][500:], c='b', label="Blue")
plt.xlim(right=1)
plt.xlabel("Input")
plt.ylabel("Degree of membership")
plt.legend(loc="best")
plt.show()

