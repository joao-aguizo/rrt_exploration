#!/usr/bin/env python

import rospy
import tf
from numpy import array, array_equal
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped
from numpy import floor
from numpy.linalg import norm
from numpy import inf


class robot:
    _position = array([.0, .0])

    def __init__(self):
        self.assigned_point = []
        self.global_frame = rospy.get_param('~global_frame', 'map')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')
        self.plan_service = rospy.get_param(
            '~plan_service', 'move_base_node/NavfnROS/make_plan')

        # transform listener for tf transformations
        self.tf_listener = tf.TransformListener()

        # initialize assigned point to unit
        self.assigned_point = self.getPosition()

        self.client = actionlib.SimpleActionClient(
            'move_base', MoveBaseAction)
        self.client.wait_for_server()

        rospy.loginfo("Done Initializing Robot!")

    def getPosition(self):
        try:
            self.tf_listener.waitForTransform(
                self.global_frame, self.robot_frame, rospy.Time(), rospy.Duration(3.0))
            (trans, _) = self.tf_listener.lookupTransform(
                self.global_frame, self.robot_frame, rospy.Time())
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, tf.Exception):
            rospy.logwarn(
                "Failed to get current robot's position! Returning last available position...")
            return self._position

        self._position = array([trans[0], trans[1]])
        return self._position

    def sendGoal(self, point, active_cb=None, done_cb=None):
        # create the move_base goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.global_frame
        goal.target_pose.pose.position.x = point[0]
        goal.target_pose.pose.position.y = point[1]
        goal.target_pose.pose.orientation.w = 1.0

        if self.getState() is actionlib.GoalStatus.ACTIVE and \
                array_equal(self.assigned_point, array(point)):
            rospy.logdebug("Following best revenue centroid...")
            return

        # send the goal to move_base action server
        self.client.send_goal(goal, active_cb=active_cb, done_cb=done_cb)
        self.assigned_point = array(point)

    def cancelGoal(self):
        self.client.cancel_goal()
        self.assigned_point = self.getPosition()

    def getState(self):
        return self.client.get_state()

    def makePlan(self, start, end):
        start = PoseStamped()
        start.header.frame_id = self.global_frame
        start.pose.position.x = start[0]
        start.pose.position.y = start[1]

        end = PoseStamped()
        end.header.frame_id = self.global_frame
        end.pose.position.x = end[0]
        end.pose.position.y = end[1]

        start = self.tf_listener.transformPose(self.global_frame, start)
        end = self.tf_listener.transformPose(self.global_frame, end)

        rospy.wait_for_service(self.plan_service)
        srv = rospy.ServiceProxy(self.plan_service, GetPlan)
        try:
            res = srv(start=start, goal=end, tolerance=0.0)
        except rospy.ServiceException as exc:
            rospy.logwarn("Service did not process request: {}".format(exc))

        return res.plan.poses
# ________________________________________________________________________________


def index_of_point(mapData, Xp):
    resolution = mapData.info.resolution
    Xstartx = mapData.info.origin.position.x
    Xstarty = mapData.info.origin.position.y
    width = mapData.info.width
    Data = mapData.data
    index = int((floor((Xp[1]-Xstarty)/resolution) *
                 width)+(floor((Xp[0]-Xstartx)/resolution)))
    return index


def point_of_index(mapData, i):
    y = mapData.info.origin.position.y + \
        (i/mapData.info.width)*mapData.info.resolution
    x = mapData.info.origin.position.x + \
        (i-(i/mapData.info.width)*(mapData.info.width))*mapData.info.resolution
    return array([x, y])
# ________________________________________________________________________________


def informationGain(mapData, point, r):
    infoGain = 0
    index = index_of_point(mapData, point)
    r_region = int(r/mapData.info.resolution)
    init_index = index-r_region*(mapData.info.width+1)
    for n in range(0, 2*r_region+1):
        start = n*mapData.info.width+init_index
        end = start+2*r_region
        limit = ((start/mapData.info.width)+2)*mapData.info.width
        for i in range(start, end+1):
            if (i >= 0 and i < limit and i < len(mapData.data)):
                if mapData.data[i] == -1:
                    infoGain += 1
    return infoGain*(mapData.info.resolution**2)
# ________________________________________________________________________________


def discount(mapData, assigned_pt, centroids, infoGain, r):
    index = index_of_point(mapData, assigned_pt)
    resolution = mapData.info.resolution
    r_region = int(r/resolution)
    init_index = index-r_region*(mapData.info.width+1)
    for n in range(0, 2*r_region+1):
        start = n*mapData.info.width+init_index
        end = start+2*r_region
        limit = ((start/mapData.info.width)+2)*mapData.info.width
        for i in range(start, end+1):
            if (i >= 0 and i < limit and i < len(mapData.data)):
                for j in range(0, len(centroids)):
                    current_pt = centroids[j]
                    if(mapData.data[i] == -1 and norm(point_of_index(mapData, i)-current_pt) <= r and norm(point_of_index(mapData, i)-assigned_pt) <= r):
                        infoGain[j] -= (mapData.info.resolution**2)
    return infoGain
# ________________________________________________________________________________


def pathCost(path):
    if (len(path) > 0):
        i = len(path)/2
        p1 = array([path[i-1].pose.position.x, path[i-1].pose.position.y])
        p2 = array([path[i].pose.position.x, path[i].pose.position.y])
        return norm(p1-p2)*(len(path)-1)
    else:
        return inf
# ________________________________________________________________________________


def unvalid(mapData, pt):
    index = index_of_point(mapData, pt)
    r_region = 5
    init_index = index-r_region*(mapData.info.width+1)
    for n in range(0, 2*r_region+1):
        start = n*mapData.info.width+init_index
        end = start+2*r_region
        limit = ((start/mapData.info.width)+2)*mapData.info.width
        for i in range(start, end+1):
            if (i >= 0 and i < limit and i < len(mapData.data)):
                if(mapData.data[i] == 1):
                    return True
    return False
# ________________________________________________________________________________


def Nearest(V, x):
    n = inf
    i = 0
    for i in range(0, V.shape[0]):
        n1 = norm(V[i, :]-x)
        if (n1 < n):
            n = n1
            result = i
    return result

# ________________________________________________________________________________


def Nearest2(V, x):
    n = inf
    for i in range(0, len(V)):
        n1 = norm(V[i]-x)

        if (n1 < n):
            n = n1
    return i
# ________________________________________________________________________________


def gridValue(mapData, Xp):
    resolution = mapData.info.resolution
    Xstartx = mapData.info.origin.position.x
    Xstarty = mapData.info.origin.position.y

    width = mapData.info.width
    Data = mapData.data
    # returns grid value at "Xp" location
    # map data:  100 occupied      -1 unknown       0 free
    index = (floor((Xp[1]-Xstarty)/resolution)*width) + \
        (floor((Xp[0]-Xstartx)/resolution))

    if int(index) < len(Data):
        return Data[int(index)]
    else:
        return 100
