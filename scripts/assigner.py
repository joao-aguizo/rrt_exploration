#!/usr/bin/env python

#--------Include modules---------------
from copy import copy
import rospy
from nav_msgs.msg import OccupancyGrid
from rrt_exploration.msg import PointArray
from numpy import array
from functions import robot, informationGain, discount
from numpy.linalg import norm
from enum import Enum


# robot status enumeration
class Status(Enum):
    IDLE = 0
    BUSY = 1


# node global variables
_map = OccupancyGrid()
_frontiers = []
_status = Status.IDLE
_revenue = float('-inf')


def points_cb(data):
    global _frontiers
    _frontiers = []
    for point in data.points:
        _frontiers.append(array([point.x, point.y]))


def map_cb(data):
    global _map
    _map = data


def active_cb():
    global _status
    _status = Status.BUSY


def done_cb(status, result):
    global _status
    _status = Status.IDLE


def node():
    global _frontiers, _map, _status, _revenue
    rospy.init_node('assigner', anonymous=False, log_level=rospy.INFO)

    # fetching all parameters
    # this can be smaller than the laser scanner range, >> smaller >>less computation time>> too small is not good, info gain won't be accurate
    info_radius = rospy.get_param('~info_radius', 1.0)
    info_multiplier = rospy.get_param('~info_multiplier', 3.0)
    # at least as much as the laser scanner range
    hysteresis_radius = rospy.get_param('~hysteresis_radius', 3.0)
    # bigger than 1 (biase robot to continue exploring current region
    hysteresis_gain = rospy.get_param('~hysteresis_gain', 2.0)
    delay_after_assignement = rospy.get_param('~delay_after_assignement', 0.5)
    rate = rospy.Rate(rospy.get_param('~rate', 100))

    rospy.Subscriber("map", OccupancyGrid, map_cb)
    rospy.Subscriber("filtered_points", PointArray, points_cb)

    # robot helper class
    bot = robot()

    while not rospy.is_shutdown():
        rate.sleep()

        # jump iteration if while no data is available
        if len(_frontiers) < 1 or len(_map.data) < 1:
            continue

        centroids = copy(_frontiers)

        # get information gain for each frontier point
        infoGain = []
        for ip in range(0, len(centroids)):
            infoGain.append(informationGain(
                _map, [centroids[ip][0], centroids[ip][1]], info_radius))

        # get dicount and update informationGain
        infoGain = discount(
            _map, bot.assigned_point, centroids, infoGain, info_radius)

        rospy.logdebug("infoGain: {}".format(infoGain))

        revenue_record = []
        centroid_record = []
        if _status is Status.IDLE:
            for ip in range(0, len(centroids)):
                cost = norm(bot.getPosition()-centroids[ip])
                information_gain = infoGain[ip]

                if cost <= hysteresis_radius:
                    information_gain *= hysteresis_gain

                revenue = information_gain*info_multiplier-cost
                revenue_record.append(revenue)
                centroid_record.append(centroids[ip])

        elif _status is Status.BUSY:
            for ip in range(0, len(centroids)):
                cost = norm(bot.getPosition()-centroids[ip])
                information_gain = infoGain[ip]

                if cost <= hysteresis_radius:
                    information_gain *= hysteresis_gain

                if norm(centroids[ip]-bot.assigned_point) < hysteresis_radius:
                    information_gain = informationGain(
                        _map, [centroids[ip][0], centroids[ip][1]], info_radius)*hysteresis_gain

                revenue = information_gain*info_multiplier-cost
                revenue_record.append(revenue)
                centroid_record.append(centroids[ip])

        # rospy.loginfo("revenue record: "+ str(revenue_record))
        # rospy.loginfo("centroid record: "+ str(centroid_record))
        # rospy.loginfo("robot IDs record: " + str(id_record))
        winner_revenue = max(revenue_record)

        rospy.loginfo("status: {} | gain: {} | cost: {} | winner: {}".format(
            str(_status), information_gain, cost, winner_revenue))

        winner_revenue_id = revenue_record.index(winner_revenue)
        bot.sendGoal(
            centroid_record[winner_revenue_id], active_cb=active_cb, done_cb=done_cb)
        rospy.sleep(delay_after_assignement)


if __name__ == '__main__':
    try:
        node()
    except rospy.ROSInterruptException:
        pass
