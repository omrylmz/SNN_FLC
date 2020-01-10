"""
This module contains the transfer function which is responsible for determining the linear twist
component of the husky's movement based on the left and right wheel neuron
"""
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
from std_msgs.msg import Float64


#This script is not necessary. It helps monitoring the monitoring wheel command.
@nrp.MapSpikeSink("right_wheel_neuron", nrp.brain.actors[1], nrp.raw_signal)
@nrp.Neuron2Robot(Topic('/right_velocity', std_msgs.msg.Float64))
def right_velocity(t, right_wheel_neuron):
    return std_msgs.msg.Float64(right_wheel_neuron.value)
