import numpy as np
import matplotlib.pyplot as plt
import nengo
from nengo.dists import Choice
from nengo.processes import Piecewise

def trimf(F, err=[1, 1, 1]):
    with nengo.Network() as TRIMF:
        TRIMF.I = nengo.Ensemble(800, dimensions=1, radius=1)
        TRIMF.M = nengo.Ensemble(400, dimensions=1, radius=0.5)
        TRIMF.O = nengo.Ensemble(800, dimensions=1, radius=1)
        TRIMF.I.encoders=Choice([[0.5], [-0.5]])

        def trimf_func(x):
            if x <= F[0]:
                return -0.5

            elif x <= F[1]:
                if (x - F[0]) / (F[1] - F[0]) < 0.01:
                    return ((x - F[0]) / (F[1] - F[0]) - 0.5) * err[0]
                elif (x - F[0]) / (F[1] - F[0]) > 0.99:
                    return ((x - F[0]) / (F[1] - F[0]) - 0.5) * err[1]
                else:
                    return (x - F[0]) / (F[1] - F[0]) - 0.5

            elif x <= F[2]:
                if (F[2] - x) / (F[2] - F[1]) > 0.99:
                    return ((F[2] - x) / (F[2] - F[1]) - 0.5) * err[1]
                elif (F[2] - x) / (F[2] - F[1]) < 0.01:
                    return ((F[2] - x) / (F[2] - F[1]) - 0.5) * err[2]
                else:
                    return (F[2] - x) / (F[2] - F[1]) - 0.5

            else:
                return -0.5

        def add_func(x):
                return 0.5 + x

        nengo.Connection(TRIMF.I, TRIMF.M, function=trimf_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
        nengo.Connection(TRIMF.M, TRIMF.O, function=add_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
        return TRIMF

model = nengo.Network(label='OA_SNN')
with model:  
    time = np.arange(0, 2.5, 0.01)
    input = time - 1.5

    stim_red = nengo.Node(Piecewise(dict(zip(time, input))))
    stim_green = nengo.Node(Piecewise(dict(zip(time, input))))
    stim_blue = nengo.Node(Piecewise(dict(zip(time, input))))

    red = trimf([-1, -0.7, -0.4], [1, 11, 6])
    nengo.Connection(stim_red, red.I)

    green = trimf([-0.4, -0.15, 0.1], [3, 10, 10])
    nengo.Connection(stim_green, green.I)

    blue = trimf([0.1, 0.5, 0.9], [1, 6, 1])
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
plt.axvspan(-0.4, 0.1, color='g', alpha=0.3)
plt.xlim(right=1)
plt.xlabel("Input")
plt.ylabel("Degree of membership")
plt.legend(loc="best")
plt.show()

