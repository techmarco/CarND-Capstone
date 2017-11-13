from pid import PID
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, *args, **kwargs):
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

        # Kp, Ki, Kd, min, max
        self.throttle_pid = PID(0.3, 0, 0, mn=0., mx=1.0)
        self.brake_pid = PID(5000, 0, 0, mn=0., mx=20000.0)
        self.steer_pid = PID(0.5, 0, 0, mn=-0.5, mx=0.5)

    def control(self, *args, **kwargs):
        # Return throttle, brake, steer
        self.plv = kwargs['proposed_linear_velocity']
        self.pav = kwargs['proposed_angular_velocity']
        self.lv = kwargs['linear_velocity']
        self.dbw_en = kwargs['dbw_enabled']

        # Simulation overrun check
        if self.plv < 0:
            rospy.logerr('Overrun error!!! Value: %f' % self.plv)
            self.plv = abs(self.plv)

        # DBW system disable (manual control)
        if self.dbw_en == False:
            self.throttle_pid.reset()
            self.brake_pid.reset()
        else:
            # Brake hold
            if abs(self.plv - self.lv) < 0.01 and self.lv > 0.0001:
                self.brake = 1
                self.throttle = 0
                self.steer = 0
            else:
            # Normal operation
                self.throttle = self.throttle_pid.step(self.plv - self.lv, 0.02)
                if self.throttle < 0.01:
                    self.brake = self.brake_pid.step(self.lv - self.plv, 0.02)
                else:
                    self.brake = 0
                self.steer = self.steer_pid.step(self.lv * self.pav, 0.02)

        return self.throttle, self.brake, self.steer
