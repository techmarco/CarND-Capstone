from yaw_controller import YawController
from pid import PID
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        self.vehicle_mass = kwargs['vehicle_mass']
        self.fuel_capacity = kwargs['fuel_capacity']
        self.fuel_capacity = kwargs['fuel_capacity']
        self.brake_deadband = kwargs['brake_deadband']
        self.decel_limit = kwargs['decel_limit']
        self.accel_limit = kwargs['accel_limit']
        self.wheel_radius = kwargs['wheel_radius']
        self.wheel_base = kwargs['wheel_base']
        self.steer_ratio = kwargs['steer_ratio']
        self.max_lat_accel = kwargs['max_lat_accel']
        self.max_steer_angle = kwargs['max_steer_angle']

        self.throttle = 0.
        self.brake = 0.
        self.steer = 0.

        self.yaw_controller = YawController(self.wheel_base,
                                            self.steer_ratio,
                                            0.,
                                            self.max_lat_accel,
                                            self.max_steer_angle)

        # Kp, Ki, Kd, min, max
        self.throttle_pid = PID(10, 1, 5, mn=0., mx=1.0)
        self.brake_pid = PID(10, 1, 5, mn=0., mx=1.0)

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        self.plv = kwargs['proposed_linear_velocity']
        self.pav = kwargs['proposed_angular_velocity']
        self.lv = kwargs['linear_velocity']
        self.dbw_en = kwargs['dbw_enabled']

        rospy.logwarn('Error between proposed linear velocity and linear velocity: %f' % (self.plv - self.lv))
        rospy.logwarn('Error between proposed angular velocity and angular velocity (0): %f' % (self.pav))

        if self.dbw_en == False:
            rospy.logerr('PID controllers reset')
            self.throttle_pid.reset()
            self.brake_pid.reset()
        else:
            self.throttle = self.throttle_pid.step(self.plv - self.lv, 50)
            self.brake = self.brake_pid.step(self.lv - self.plv, 50)

        self.steer = self.yaw_controller.get_steering(self.lv, self.pav, self.lv)

        rospy.logwarn('Throttle: %f, Brake: %f, Steer: %f' % (self.throttle, self.brake, self.steer))

        return self.throttle, self.brake, self.steer
