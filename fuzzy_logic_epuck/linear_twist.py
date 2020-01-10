"""
This module contains the transfer function which is responsible for determining the linear twist
component of the husky's movement based on the left and right wheel neuron
"""
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg


@nrp.MapSpikeSink("left_wheel_neuron", nrp.brain.actors[0], nrp.raw_signal)
@nrp.MapSpikeSink("right_wheel_neuron", nrp.brain.actors[1], nrp.raw_signal)
@nrp.Neuron2Robot(Topic('/epuck/cmd_vel', geometry_msgs.msg.Twist))
def linear_twist(t, left_wheel_neuron, right_wheel_neuron):
    """
    The transfer function which calculates the linear twist of the husky robot based on the
    voltage of left and right wheel neuron.

    :param t: the current simulation time
    :param left_wheel_neuron: the left wheel neuron device
    :param right_wheel_neuron: the right whesel neuron device
    :return: a geometry_msgs/Twist message setting the linear twist fo the husky robot movement.
    """

    return geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(
        x=1*(right_wheel_neuron.value + left_wheel_neuron.value), y=0.0, z=0.0),
        angular=geometry_msgs.msg.Vector3(x=0.0, y=0.0, 
            z=3*(right_wheel_neuron.value - left_wheel_neuron.value)))
