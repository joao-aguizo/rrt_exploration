#!/usr/bin/env python

import tf
import rospy
from copy import copy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped
from numpy import array, vstack, delete
from functions import gridValue, informationGain
from sklearn.cluster import MeanShift
from rrt_exploration.msg import PointArray

_map = OccupancyGrid()
_costmap = OccupancyGrid()
_frontiers = []


def point_cb(data, args):
    global _map, _frontiers

    # only if map data exists
    if len(_map.data) < 1:
        return

    # get map frame
    frame = _map.header.frame_id

    args[0].waitForTransform(
        frame, args[1], rospy.Time(), rospy.Duration(3.0))
    transformedPoint = args[0].transformPoint(frame, data)
    x = [array([transformedPoint.point.x, transformedPoint.point.y])]
    if len(_frontiers) > 0:
        _frontiers = vstack((_frontiers, x))
    else:
        _frontiers = x


def map_cb(data):
    global _map
    _map = data


def costmap_cb(data):
    global _costmap
    _costmap = data


def node():
    global _map, _costmap, _frontiers
    rospy.init_node('filter', anonymous=False)

    # fetch all parameters
    threshold = rospy.get_param('~costmap_clearing_threshold', 70)
    # this can be smaller than the laser scanner range, >> smaller >>less computation time>> too small is not good, info gain won't be accurate
    info_radius = rospy.get_param('~info_radius', 1.0)
    robot_frame = rospy.get_param('~robot_frame', 'base_link')
    rate = rospy.Rate(rospy.get_param('~rate', 100))

    # subscribes to a map and costmap
    rospy.Subscriber("map", OccupancyGrid, map_cb)
    rospy.Subscriber("costmap", OccupancyGrid, costmap_cb)

    # detected points subscriber (uses tf listener and robot's frame)
    listener = tf.TransformListener()
    rospy.Subscriber("detected_points", PointStamped, callback=point_cb,
                     callback_args=[listener, robot_frame])

    frontiers_pub = rospy.Publisher('frontiers', Marker, queue_size=10)
    centroids_pub = rospy.Publisher('centroids', Marker, queue_size=10)
    points_pub = rospy.Publisher('filtered_points', PointArray, queue_size=10)

    while not rospy.is_shutdown():
        rate.sleep()

        # jump iteration if while no data is available
        if len(_frontiers) < 1 or len(_map.data) < 1 or len(_costmap.data) < 1:
            rospy.loginfo_once("Waiting for inputs...")
            continue

        rospy.loginfo("Got them!")

        # clustering frontier points
        centroids = []
        tmp = copy(_frontiers)
        if len(tmp) > 1:
            ms = MeanShift(bandwidth=0.3)  # TODO: parameterize this
            ms.fit(tmp)
            centroids = ms.cluster_centers_  # centroids array is the centers of each cluster
        else:  # len(tmp) == 1, considering the first condition of the loop
            # if there is only one frontier no need for clustering, i.e. centroids=frontiers
            centroids = tmp

        # TODO: why override original variable?!?!?!
        # _frontiers = copy(centroids)

        # clearing old frontiers
        z = 0
        while z < len(centroids):
            tmp = PointStamped()
            tmp.header.frame_id = _map.header.frame_id
            tmp.point.x = centroids[z][0]
            tmp.point.y = centroids[z][1]
            tmp.point.z = .0

            transformedPoint = listener.transformPoint(
                _costmap.header.frame_id, tmp)

            x = array([transformedPoint.point.x, transformedPoint.point.y])

            if gridValue(_costmap, x) > threshold or \
                    (informationGain(_map, [centroids[z][0], centroids[z][1]], info_radius*0.5)) < 0.2:
                centroids = delete(centroids, (z), axis=0)
                z = z-1

            z += 1

        ###########################
        #### PUBLISH MESSAGES #####
        ###########################

        # assemble filtered points message
        point_array = PointArray()
        for i in centroids:
            point_array.points.append(Point(x=i[0], y=i[1], z=.0))

        # publish filtered points
        points_pub.publish(point_array)

        # assemble frontiers marker message
        marker = Marker()
        marker.header.frame_id = _map.header.frame_id
        marker.ns = "markers2"
        marker.type = Marker.POINTS
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.color.r = 255.0/255.0
        marker.color.g = 255.0/255.0
        marker.color.b = 0.0/255.0
        marker.color.a = 1

        for c in centroids:
            marker.points.append(Point(x=c[0], y=c[1], z=.0))

        # publish frontiers
        frontiers_pub.publish(marker)

        # re-use same marker message (TODO: possibly needs copying!)
        marker.ns = "markers3"
        marker.id = 4
        marker.color.r = 0.0/255.0
        marker.points = []

        for c in centroids:
            marker.points.append(Point(x=c[0], y=c[1], z=.0))

        # publish centroids
        centroids_pub.publish(marker)


if __name__ == '__main__':
    try:
        node()
    except rospy.ROSInterruptException:
        pass
