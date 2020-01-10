import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import logging

@nrp.MapRobotSubscriber("dist0", Topic('/epuck/distanceSensor0', sensor_msgs.msg.Range))
@nrp.MapRobotSubscriber("dist1", Topic('/epuck/distanceSensor1', sensor_msgs.msg.Range))
@nrp.MapRobotSubscriber("dist2", Topic('/epuck/distanceSensor2', sensor_msgs.msg.Range))
@nrp.MapRobotSubscriber("dist3", Topic('/epuck/distanceSensor3', sensor_msgs.msg.Range))
@nrp.MapRobotSubscriber("dist4", Topic('/epuck/distanceSensor4', sensor_msgs.msg.Range))
@nrp.MapRobotSubscriber("dist5", Topic('/epuck/distanceSensor5', sensor_msgs.msg.Range))
@nrp.MapRobotSubscriber("dist6", Topic('/epuck/distanceSensor6', sensor_msgs.msg.Range))
@nrp.MapRobotSubscriber("dist7", Topic('/epuck/distanceSensor7', sensor_msgs.msg.Range))

@nrp.MapSpikeSource("sensor_neurons", nrp.brain.sensors, nrp.raw_signal)

@nrp.Robot2Neuron()
def distances(t, dist0, dist1, dist2, dist3, dist4, dist5, dist6, dist7, sensor_neurons):
    """
    This transfer function detects the distance values taken from the epuck and direct them to brain
    """
    
    if dist5.value is not None and dist6.value is not None:
        sensor_neurons.value[0] = min(dist5.value.range, dist6.value.range) * 2 - 1

    if dist0.value is not None and dist7.value is not None:
        sensor_neurons.value[1] = min(dist0.value.range, dist7.value.range) * 2 - 1

    if dist1.value is not None and dist2.value is not None:
        sensor_neurons.value[2] = min(dist1.value.range, dist2.value.range) * 2 - 1
