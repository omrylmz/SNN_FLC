"""
This module contains the transfer function which is responsible for determining the linear twist
component of the husky's movement based on the left and right wheel neuron
"""
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
from std_msgs.msg import Float64


#This script is not necessary. It helps monitoring the left wheel command.
@nrp.MapSpikeSink("left_wheel_neuron", nrp.brain.actors[0], nrp.raw_signal)
@nrp.Neuron2Robot(Topic('/left_velocity', std_msgs.msg.Float64))
def left_velocity(t, left_wheel_neuron):
    return std_msgs.msg.Float64(left_wheel_neuron.value)
