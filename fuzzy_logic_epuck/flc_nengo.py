import nengo
from nengo.processes import Piecewise
from nengo.dists import Choice
import numpy as np
import rospy
from std_msgs.msg import Float64


def PROD_AND(left_weight, right_weight):
    with nengo.Network() as AND:
        AND.I = nengo.Ensemble(800, dimensions=2, radius=1.6)
        AND.M = nengo.Ensemble(800, dimensions=2, radius=1.6)
        #AND.W_ = nengo.Ensemble(100, dimensions=1, radius=1)
        AND.W = nengo.Ensemble(400, dimensions=1, radius=1.1)
        AND.WR = nengo.Ensemble(400, dimensions=1, radius=1.1)
        AND.WL = nengo.Ensemble(400, dimensions=1, radius=1.1)
        AND.L = nengo.Ensemble(800, dimensions=1, radius=1)
        AND.R = nengo.Ensemble(800, dimensions=1, radius=1)

        def and_func(x):
            x_arr = np.array(x)
            return np.prod(x_arr)

        def multiply_left_func(x):
            return x * left_weight

        def multiply_right_func(x):
            return x * right_weight

        # def noise_canceller(x):
        #     return 0 if np.abs(x) < 0.02 else x

        nengo.Connection(AND.I, AND.M[0], function=and_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
        nengo.Connection(AND.M, AND.W, function=and_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
        # nengo.Connection(AND.W_, AND.W, function=noise_canceller, learning_rule_type=nengo.PES(learning_rate=2e-4))
        nengo.Connection(AND.W, AND.WL)
        nengo.Connection(AND.WL, AND.L, function=multiply_left_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
        nengo.Connection(AND.W, AND.WR)
        nengo.Connection(AND.WR, AND.R, function=multiply_right_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return AND

def tramf(F, err=[1, 1, 1, 1]):
    with nengo.Network() as tramf:
        tramf.I = nengo.Ensemble(800, dimensions=1, radius=1)
        tramf.M = nengo.Ensemble(400, dimensions=1, radius=0.5)
        tramf.O = nengo.Ensemble(800, dimensions=1, radius=1)
        tramf.I.encoders=Choice([[0.5], [-0.5]])

        def tramf_func(x):
            if x <= F[0]:
                return -0.5

            elif x <= F[1]:
                if (x - F[0]) / (F[1] - F[0]) < 0.02:
                    return ((x - F[0]) / (F[1] - F[0]) - 0.5) * err[0]
                elif (x - F[0]) / (F[1] - F[0]) > 0.98:
                    return ((x - F[0]) / (F[1] - F[0]) - 0.5) * err[1]
                else:
                    return (x - F[0]) / (F[1] - F[0]) - 0.5

            elif x <= F[2]:
                return 0.5

            elif x <= F[3]:
                if (F[3] - x) / (F[3] - F[2]) > 0.98:
                    return ((F[3] - x) / (F[3] - F[2]) - 0.5) * err[2]
                elif (F[3] - x) / (F[3] - F[2]) < 0.02:
                    return ((F[3] - x) / (F[3] - F[2]) - 0.5) * err[3]
                else:
                    return (F[3] - x) / (F[3] - F[2]) - 0.5

            else:
                return -0.5

        def add_func(x):
                return 0.5 + x

        nengo.Connection(tramf.I, tramf.M, function=tramf_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
        nengo.Connection(tramf.M, tramf.O, function=add_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
        return tramf


circuit = nengo.Network(label='OAFLC')
with circuit:
    sensors = nengo.Ensemble(n_neurons=1000, dimensions=3)

    distances = nengo.Ensemble(1600, dimensions=3)
    nengo.Connection(sensors, distances)

    Left_Motor = nengo.Ensemble(800, dimensions=1, radius=1.1)
    Right_Motor = nengo.Ensemble(800, dimensions=1, radius=1.1)


    LN = tramf([-100, -1, -0.5, -0.3], [1, 1, 5.5, 6])
    nengo.Connection(distances[0], LN.I)

    LM = tramf([-0.5, -0.3, -0.14, 0], [10, 5, 10, 5])
    nengo.Connection(distances[0], LM.I)

    LF = tramf([-0.14, 0, 1, 100], [10, 5.5, 1, 1])
    nengo.Connection(distances[0], LF.I)


    FN = tramf([-100, -1, -0.5, -0.3], [1, 1, 5.5, 6])
    nengo.Connection(distances[1], FN.I)

    FM = tramf([-0.5, -0.3, -0.14, 0], [10, 5, 10, 5])
    nengo.Connection(distances[1], FM.I)

    FF = tramf([-0.14, 0, 1, 100], [10, 5.5, 1, 1])
    nengo.Connection(distances[1], FF.I)


    RN = tramf([-100, -1, -0.5, -0.3], [1, 1, 5.5, 6])
    nengo.Connection(distances[2], RN.I)

    RM = tramf([-0.5, -0.3, -0.14, 0], [10, 5, 10, 5])
    nengo.Connection(distances[2], RM.I)

    RF = tramf([-0.14, 0, 1, 100], [10, 5.5, 1, 1])
    nengo.Connection(distances[2], RF.I)

    HN = -0.5
    N = -0.25
    Z = 0
    P = 0.25
    HP = 0.5
    VHP = 1


    Rule_LN_x_FN_x_RN = PROD_AND(HN, HN)
    nengo.Connection(LN.O, Rule_LN_x_FN_x_RN.I[0])
    nengo.Connection(FN.O, Rule_LN_x_FN_x_RN.I[1])
    nengo.Connection(RN.O, Rule_LN_x_FN_x_RN.M[1])
    nengo.Connection(Rule_LN_x_FN_x_RN.L, Left_Motor)
    nengo.Connection(Rule_LN_x_FN_x_RN.R, Right_Motor)

    Rule_LN_x_FN_x_RM = PROD_AND(HP, HN)
    nengo.Connection(LN.O, Rule_LN_x_FN_x_RM.I[0])
    nengo.Connection(FN.O, Rule_LN_x_FN_x_RM.I[1])
    nengo.Connection(RM.O, Rule_LN_x_FN_x_RM.M[1])
    nengo.Connection(Rule_LN_x_FN_x_RM.L, Left_Motor)
    nengo.Connection(Rule_LN_x_FN_x_RM.R, Right_Motor)

    Rule_LN_x_FN_x_RF = PROD_AND(HP, HN)
    nengo.Connection(LN.O, Rule_LN_x_FN_x_RF.I[0])
    nengo.Connection(FN.O, Rule_LN_x_FN_x_RF.I[1])
    nengo.Connection(RF.O, Rule_LN_x_FN_x_RF.M[1])
    nengo.Connection(Rule_LN_x_FN_x_RF.L, Left_Motor)
    nengo.Connection(Rule_LN_x_FN_x_RF.R, Right_Motor)


    Rule_LN_x_FM_x_RN = PROD_AND(N, HN)
    nengo.Connection(LN.O, Rule_LN_x_FM_x_RN.I[0])
    nengo.Connection(FM.O, Rule_LN_x_FM_x_RN.I[1])
    nengo.Connection(RN.O, Rule_LN_x_FM_x_RN.M[1])
    nengo.Connection(Rule_LN_x_FM_x_RN.L, Left_Motor)
    nengo.Connection(Rule_LN_x_FM_x_RN.R, Right_Motor)

    Rule_LN_x_FM_x_RM = PROD_AND(VHP, Z)
    nengo.Connection(LN.O, Rule_LN_x_FM_x_RM.I[0])
    nengo.Connection(FM.O, Rule_LN_x_FM_x_RM.I[1])
    nengo.Connection(RM.O, Rule_LN_x_FM_x_RM.M[1])
    nengo.Connection(Rule_LN_x_FM_x_RM.L, Left_Motor)
    nengo.Connection(Rule_LN_x_FM_x_RM.R, Right_Motor)

    Rule_LN_x_FM_x_RF = PROD_AND(P, Z)
    nengo.Connection(LN.O, Rule_LN_x_FM_x_RF.I[0])
    nengo.Connection(FM.O, Rule_LN_x_FM_x_RF.I[1])
    nengo.Connection(RF.O, Rule_LN_x_FM_x_RF.M[1])
    nengo.Connection(Rule_LN_x_FM_x_RF.L, Left_Motor)
    nengo.Connection(Rule_LN_x_FM_x_RF.R, Right_Motor)


    Rule_LN_x_FF_x_RN = PROD_AND(P, P)
    nengo.Connection(LN.O, Rule_LN_x_FF_x_RN.I[0])
    nengo.Connection(FF.O, Rule_LN_x_FF_x_RN.I[1])
    nengo.Connection(RN.O, Rule_LN_x_FF_x_RN.M[1])
    nengo.Connection(Rule_LN_x_FF_x_RN.L, Left_Motor)
    nengo.Connection(Rule_LN_x_FF_x_RN.R, Right_Motor)

    Rule_LN_x_FF_x_RM = PROD_AND(P, N)
    nengo.Connection(LN.O, Rule_LN_x_FF_x_RM.I[0])
    nengo.Connection(FF.O, Rule_LN_x_FF_x_RM.I[1])
    nengo.Connection(RM.O, Rule_LN_x_FF_x_RM.M[1])
    nengo.Connection(Rule_LN_x_FF_x_RM.L, Left_Motor)
    nengo.Connection(Rule_LN_x_FF_x_RM.R, Right_Motor)

    Rule_LN_x_FF_x_RF = PROD_AND(HP, Z)
    nengo.Connection(LN.O, Rule_LN_x_FF_x_RF.I[0])
    nengo.Connection(FF.O, Rule_LN_x_FF_x_RF.I[1])
    nengo.Connection(RF.O, Rule_LN_x_FF_x_RF.M[1])
    nengo.Connection(Rule_LN_x_FF_x_RF.L, Left_Motor)
    nengo.Connection(Rule_LN_x_FF_x_RF.R, Right_Motor)


    Rule_LM_x_FN_x_RN = PROD_AND(HN, HP)
    nengo.Connection(LM.O, Rule_LM_x_FN_x_RN.I[0])
    nengo.Connection(FN.O, Rule_LM_x_FN_x_RN.I[1])
    nengo.Connection(RN.O, Rule_LM_x_FN_x_RN.M[1])
    nengo.Connection(Rule_LM_x_FN_x_RN.L, Left_Motor)
    nengo.Connection(Rule_LM_x_FN_x_RN.R, Right_Motor)

    Rule_LM_x_FN_x_RM = PROD_AND(HN, HN)
    nengo.Connection(LM.O, Rule_LM_x_FN_x_RM.I[0])
    nengo.Connection(FN.O, Rule_LM_x_FN_x_RM.I[1])
    nengo.Connection(RM.O, Rule_LM_x_FN_x_RM.M[1])
    nengo.Connection(Rule_LM_x_FN_x_RM.L, Left_Motor)
    nengo.Connection(Rule_LM_x_FN_x_RM.R, Right_Motor)

    Rule_LM_x_FN_x_RF = PROD_AND(HP, N)
    nengo.Connection(LM.O, Rule_LM_x_FN_x_RF.I[0])
    nengo.Connection(FN.O, Rule_LM_x_FN_x_RF.I[1])
    nengo.Connection(RF.O, Rule_LM_x_FN_x_RF.M[1])
    nengo.Connection(Rule_LM_x_FN_x_RF.L, Left_Motor)
    nengo.Connection(Rule_LM_x_FN_x_RF.R, Right_Motor)


    Rule_LM_x_FM_x_RN = PROD_AND(P, VHP)
    nengo.Connection(LM.O, Rule_LM_x_FM_x_RN.I[0])
    nengo.Connection(FM.O, Rule_LM_x_FM_x_RN.I[1])
    nengo.Connection(RN.O, Rule_LM_x_FM_x_RN.M[1])
    nengo.Connection(Rule_LM_x_FM_x_RN.L, Left_Motor)
    nengo.Connection(Rule_LM_x_FM_x_RN.R, Right_Motor)

    Rule_LM_x_FM_x_RM = PROD_AND(HP, Z)
    nengo.Connection(LM.O, Rule_LM_x_FM_x_RM.I[0])
    nengo.Connection(FM.O, Rule_LM_x_FM_x_RM.I[1])
    nengo.Connection(RM.O, Rule_LM_x_FM_x_RM.M[1])
    nengo.Connection(Rule_LM_x_FM_x_RM.L, Left_Motor)
    nengo.Connection(Rule_LM_x_FM_x_RM.R, Right_Motor)

    Rule_LM_x_FM_x_RF = PROD_AND(VHP, P)
    nengo.Connection(LM.O, Rule_LM_x_FM_x_RF.I[0])
    nengo.Connection(FM.O, Rule_LM_x_FM_x_RF.I[1])
    nengo.Connection(RF.O, Rule_LM_x_FM_x_RF.M[1])
    nengo.Connection(Rule_LM_x_FM_x_RF.L, Left_Motor)
    nengo.Connection(Rule_LM_x_FM_x_RF.R, Right_Motor)


    Rule_LM_x_FF_x_RN = PROD_AND(Z, HP)
    nengo.Connection(LM.O, Rule_LM_x_FF_x_RN.I[0])
    nengo.Connection(FF.O, Rule_LM_x_FF_x_RN.I[1])
    nengo.Connection(RN.O, Rule_LM_x_FF_x_RN.M[1])
    nengo.Connection(Rule_LM_x_FF_x_RN.L, Left_Motor)
    nengo.Connection(Rule_LM_x_FF_x_RN.R, Right_Motor)

    Rule_LM_x_FF_x_RM = PROD_AND(HP, N)
    nengo.Connection(LM.O, Rule_LM_x_FF_x_RM.I[0])
    nengo.Connection(FF.O, Rule_LM_x_FF_x_RM.I[1])
    nengo.Connection(RM.O, Rule_LM_x_FF_x_RM.M[1])
    nengo.Connection(Rule_LM_x_FF_x_RM.L, Left_Motor)
    nengo.Connection(Rule_LM_x_FF_x_RM.R, Right_Motor)

    Rule_LM_x_FF_x_RF = PROD_AND(VHP, P)
    nengo.Connection(LM.O, Rule_LM_x_FF_x_RF.I[0])
    nengo.Connection(FF.O, Rule_LM_x_FF_x_RF.I[1])
    nengo.Connection(RF.O, Rule_LM_x_FF_x_RF.M[1])
    nengo.Connection(Rule_LM_x_FF_x_RF.L, Left_Motor)
    nengo.Connection(Rule_LM_x_FF_x_RF.R, Right_Motor)


    Rule_LF_x_FN_x_RN = PROD_AND(HN, HP)
    nengo.Connection(LF.O, Rule_LF_x_FN_x_RN.I[0])
    nengo.Connection(FN.O, Rule_LF_x_FN_x_RN.I[1])
    nengo.Connection(RN.O, Rule_LF_x_FN_x_RN.M[1])
    nengo.Connection(Rule_LF_x_FN_x_RN.L, Left_Motor)
    nengo.Connection(Rule_LF_x_FN_x_RN.R, Right_Motor)

    Rule_LF_x_FN_x_RM = PROD_AND(N, HP)
    nengo.Connection(LF.O, Rule_LF_x_FN_x_RM.I[0])
    nengo.Connection(FN.O, Rule_LF_x_FN_x_RM.I[1])
    nengo.Connection(RM.O, Rule_LF_x_FN_x_RM.M[1])
    nengo.Connection(Rule_LF_x_FN_x_RM.L, Left_Motor)
    nengo.Connection(Rule_LF_x_FN_x_RM.R, Right_Motor)

    Rule_LF_x_FN_x_RF = PROD_AND(N, HN)
    nengo.Connection(LF.O, Rule_LF_x_FN_x_RF.I[0])
    nengo.Connection(FN.O, Rule_LF_x_FN_x_RF.I[1])
    nengo.Connection(RF.O, Rule_LF_x_FN_x_RF.M[1])
    nengo.Connection(Rule_LF_x_FN_x_RF.L, Left_Motor)
    nengo.Connection(Rule_LF_x_FN_x_RF.R, Right_Motor)


    Rule_LF_x_FM_x_RN = PROD_AND(Z, P)
    nengo.Connection(LF.O, Rule_LF_x_FM_x_RN.I[0])
    nengo.Connection(FM.O, Rule_LF_x_FM_x_RN.I[1])
    nengo.Connection(RN.O, Rule_LF_x_FM_x_RN.M[1])
    nengo.Connection(Rule_LF_x_FM_x_RN.L, Left_Motor)
    nengo.Connection(Rule_LF_x_FM_x_RN.R, Right_Motor)

    Rule_LF_x_FM_x_RM = PROD_AND(P, VHP)
    nengo.Connection(LF.O, Rule_LF_x_FM_x_RM.I[0])
    nengo.Connection(FM.O, Rule_LF_x_FM_x_RM.I[1])
    nengo.Connection(RM.O, Rule_LF_x_FM_x_RM.M[1])
    nengo.Connection(Rule_LF_x_FM_x_RM.L, Left_Motor)
    nengo.Connection(Rule_LF_x_FM_x_RM.R, Right_Motor)

    Rule_LF_x_FM_x_RF = PROD_AND(VHP, P)
    nengo.Connection(LF.O, Rule_LF_x_FM_x_RF.I[0])
    nengo.Connection(FM.O, Rule_LF_x_FM_x_RF.I[1])
    nengo.Connection(RF.O, Rule_LF_x_FM_x_RF.M[1])
    nengo.Connection(Rule_LF_x_FM_x_RF.L, Left_Motor)
    nengo.Connection(Rule_LF_x_FM_x_RF.R, Right_Motor)


    Rule_LF_x_FF_x_RN = PROD_AND(N, P)
    nengo.Connection(LF.O, Rule_LF_x_FF_x_RN.I[0])
    nengo.Connection(FF.O, Rule_LF_x_FF_x_RN.I[1])
    nengo.Connection(RN.O, Rule_LF_x_FF_x_RN.M[1])
    nengo.Connection(Rule_LF_x_FF_x_RN.L, Left_Motor)
    nengo.Connection(Rule_LF_x_FF_x_RN.R, Right_Motor)

    Rule_LF_x_FF_x_RM = PROD_AND(P, VHP)
    nengo.Connection(LF.O, Rule_LF_x_FF_x_RM.I[0])
    nengo.Connection(FF.O, Rule_LF_x_FF_x_RM.I[1])
    nengo.Connection(RM.O, Rule_LF_x_FF_x_RM.M[1])
    nengo.Connection(Rule_LF_x_FF_x_RM.L, Left_Motor)
    nengo.Connection(Rule_LF_x_FF_x_RM.R, Right_Motor)

    Rule_LF_x_FF_x_RF = PROD_AND(HP, HP)
    nengo.Connection(LF.O, Rule_LF_x_FF_x_RF.I[0])
    nengo.Connection(FF.O, Rule_LF_x_FF_x_RF.I[1])
    nengo.Connection(RF.O, Rule_LF_x_FF_x_RF.M[1])
    nengo.Connection(Rule_LF_x_FF_x_RF.L, Left_Motor)
    nengo.Connection(Rule_LF_x_FF_x_RF.R, Right_Motor)

    actors = nengo.Ensemble(n_neurons=2, dimensions=2, neuron_type=nengo.Direct())
    
    nengo.Connection(Left_Motor, actors[0])
    nengo.Connection(Right_Motor, actors[1])
