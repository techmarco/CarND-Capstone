#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=rospy.INFO)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.wp_idx = None
        self.last_position = None
        self.pose = None
        self.waypoints = None
        self.waypoints_count = 0
        self.seq = 0
        self.stopping = False
        self.stopped = False
        self.stop_complete = False
        self.lights = []
        self.old_traffic = -2
        self.last_wp_idx = None

        self.max_velocity = self.kmph2mps(3*11.11) # 33.33 Km/h
        rospy.logwarn('Maximum simulator velocity (mps): %f' % self.max_velocity)

        # Final waypoints publisher
        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        rospy.spin()

    def pose_cb(self, msg):
        time = rospy.Time.now()
        if self.wp_idx is not None:
            if self.wp_idx % 25 == 0 and self.wp_idx != self.last_wp_idx:
                rospy.logwarn('~~~ Current waypint: %d ...' % self.wp_idx)
            self.last_wp_idx = self.wp_idx

        # determine waypoint closest to current pose
        self.pose = msg

        wp_amin = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        dist = []
        if self.waypoints is not None:
            for wp in self.waypoints:
                dist.append(dl(wp.pose.pose.position, msg.pose.position))
            wp_amin = np.argmin(dist)
            self.wp_idx = (wp_amin + 1) % self.waypoints_count

            if self.stopping or self.stopped:
                return

            lane = Lane()
            lane.header.frame_id = '/world_pose'
            lane.header.stamp = rospy.Time.now()
            lane.header.seq = self.seq

            range_list = range(self.wp_idx, (self.wp_idx + LOOKAHEAD_WPS))
            range_list = [i % self.waypoints_count for i in range(self.wp_idx, self.wp_idx + LOOKAHEAD_WPS)]

            start_vel = self.max_velocity
            end_vel = self.max_velocity
            velocities_lin = np.linspace(start_vel, end_vel, num=len(range_list))

            idx = 0
            for i in range_list:
                self.set_waypoint_velocity(self.waypoints, i, velocities_lin[idx])
                idx += 1

            lane.waypoints = [self.waypoints[i] for i in range_list]
            self.final_waypoints_pub.publish(lane)
            self.seq += 1


    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints
        self.waypoints_count = len(self.waypoints)
        idx = 0
        for i in range(self.waypoints_count):
            self.set_waypoint_velocity(self.waypoints, i, self.max_velocity)
            idx += 1


    def traffic_cb(self, msg):
        if self.old_traffic == msg.data and self.stop_complete:
            # rospy.logwarn('Ignoring old light due to proximity...')
            self.stopped = False
            self.stopping = False
            return

        if msg.data == -1:
            if self.stopped == True:
                rospy.logwarn('~~~~~~~~~~~~~~~~~~~ Releasing brakes ...')
                self.stopping = False
                self.stopped = False
                self.stop_complete = True
            elif self.stopping == True:
                rospy.logwarn('~~~~~~~~~~~~~~~~~~~ Rolling stop ....')
                self.stopping = False
                self.stop_complete = True
            return

        if self.stopped == True:
            return

        range_list = [i % self.waypoints_count for i in range(self.wp_idx, self.wp_idx + LOOKAHEAD_WPS)]
        zeros_after = abs(msg.data - 5 - self.wp_idx) % self.waypoints_count

        start_vel = self.get_waypoint_velocity(self.waypoints[self.wp_idx])
        if start_vel == 0:
            self.stopped = True
        end_vel = 0

        velocities = end_vel * np.ones(len(range_list))
        velocities_log = np.logspace(start_vel, end_vel, num=zeros_after, base=10)
        velocities[0:zeros_after] = np.log(velocities_log) / np.log(10)

        idx = 0
        for i in range_list:
            self.set_waypoint_velocity(self.waypoints, i, velocities[idx])
            idx += 1

        lane = Lane()
        lane.header.frame_id = '/world_brake'
        lane.header.stamp = rospy.Time.now()
        lane.header.seq = self.seq
        lane.waypoints = [self.waypoints[i] for i in range_list]
        self.final_waypoints_pub.publish(lane)

        self.seq += 1
        if msg.data - self.old_traffic != 0:
            self.old_traffic = msg.data
            self.stop_complete = False
        self.stopping = True


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity
        waypoints[waypoint].twist.header.seq = waypoint

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def kmph2mps(self, velocity_kmph):
        return (velocity_kmph * 1000.) / (60. * 60.)

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
