#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include "stdint.h"
#include "functions.h"
#include "mtrand.h"

#include "nav_msgs/OccupancyGrid.h"
#include "geometry_msgs/PointStamped.h"
#include "std_msgs/Header.h"
#include "nav_msgs/MapMetaData.h"
#include "geometry_msgs/Point.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include <tf/transform_listener.h>

// global variables
nav_msgs::OccupancyGrid mapData;
geometry_msgs::PointStamped exploration_goal;
float xdim, ydim, resolution, Xstartx, Xstarty, init_map_x, init_map_y;
std::vector<geometry_msgs::Point> userDefinedFrontiers;
bool userDefinedFrontiersInitialized = false;
rdm r; // for generating random numbers
visualization_msgs::Marker lines, points;

void mapCallBack(const nav_msgs::OccupancyGrid::ConstPtr &msg)
{
	mapData = *msg;
}

void rvizCallBack(const geometry_msgs::PointStamped::ConstPtr &msg)
{
	if (userDefinedFrontiers.size() > 4)
	{
		userDefinedFrontiersInitialized = false;
		userDefinedFrontiers.clear();
		ROS_INFO("Restart user defined frontier acquisition through RVIZ.");
	}

	userDefinedFrontiers.push_back(msg->point);
}

int main(int argc, char **argv)
{

	unsigned long init[4] = {0x123, 0x234, 0x345, 0x456}, length = 7;
	MTRand_int32 irand(init, length); // 32-bit int generator
									  // this is an example of initializing by an array
									  // you may use MTRand(seed) with any 32bit integer
									  // as a seed for a simpler initialization
	MTRand drand;					  // double in [0, 1) generator, already init

	// generate the same numbers as in the original C test program
	ros::init(argc, argv, "global_rrt_frontier_detector");
	ros::NodeHandle nh;

	// parameters variables
	float rateHz, eta;

	// parameters
	ros::param::param<float>("~eta", eta, 0.5);
	ros::param::param<float>("~rate", rateHz, 100);

	// subscribers
	ros::Subscriber map_sub = nh.subscribe("map", 100, mapCallBack);
	ros::Subscriber rviz_sub = nh.subscribe("/clicked_point", 100, rvizCallBack);

	// publishers
	ros::Publisher point_pub = nh.advertise<geometry_msgs::PointStamped>("detected_points", 10);
	ros::Publisher marker_pub = nh.advertise<visualization_msgs::MarkerArray>(ros::this_node::getName() + "_shapes", 10);

	// algorithm variables
	float xr, yr, init_map_x, init_map_y, range;
	std::vector<float> x_rand, x_nearest, x_new;
	std::vector<std::vector<float>> V;

	ros::Rate rate(rateHz);
	while (ros::ok())
	{
		ros::spinOnce();
		rate.sleep();

		// jump iteration if no data is available
		if (mapData.header.seq < 1 || mapData.data.size() < 1)
		{
			continue;
		}

		visualization_msgs::MarkerArray marker_array;

		// visualization markers setup
		points.header.frame_id = mapData.header.frame_id;
		points.ns = "points";
		points.type = points.POINTS;
		points.action = points.ADD;
		points.pose.orientation.w = 1.0;
		points.scale.x = 0.3;
		points.scale.y = 0.3;
		points.color.r = 255.0 / 255.0;
		points.color.g = 0.0 / 255.0;
		points.color.b = 0.0 / 255.0;
		points.color.a = 1.0;

		auto tmp = lines.points;
		lines = points;
		lines.points = tmp;
		lines.ns = "lines";
		lines.type = lines.LINE_LIST;
		lines.action = lines.ADD;
		lines.scale.x = 0.03;
		lines.scale.y = 0.03;
		lines.color.r = 9.0 / 255.0;
		lines.color.g = 91.0 / 255.0;
		lines.color.b = 236.0 / 255.0;

		// user defined area was set and system was not yet initialized
		if (userDefinedFrontiers.size() > 4 && !userDefinedFrontiersInitialized)
		{
			ROS_INFO_ONCE("Initializing...");

			std::vector<float> tmp_1, tmp_2, tmp_3;
			tmp_1.push_back(userDefinedFrontiers[0].x);
			tmp_1.push_back(userDefinedFrontiers[0].y);

			tmp_2.push_back(userDefinedFrontiers[2].x);
			tmp_2.push_back(userDefinedFrontiers[0].y);

			init_map_x = Norm(tmp_1, tmp_2);
			tmp_1.clear();
			tmp_2.clear();

			tmp_1.push_back(userDefinedFrontiers[0].x);
			tmp_1.push_back(userDefinedFrontiers[0].y);

			tmp_2.push_back(userDefinedFrontiers[0].x);
			tmp_2.push_back(userDefinedFrontiers[2].y);

			init_map_y = Norm(tmp_1, tmp_2);
			tmp_1.clear();
			tmp_2.clear();

			Xstartx = (userDefinedFrontiers[0].x + userDefinedFrontiers[2].x) * .5;
			Xstarty = (userDefinedFrontiers[0].y + userDefinedFrontiers[2].y) * .5;

			tmp_3.push_back(userDefinedFrontiers[4].x);
			tmp_3.push_back(userDefinedFrontiers[4].y);
			V.push_back(tmp_3);

			userDefinedFrontiersInitialized = true;
		}
		else if (userDefinedFrontiersInitialized)
		{
			ROS_INFO_ONCE("Generating trees...");

			// Sample free
			x_rand.clear();
			xr = (drand() * init_map_x) - (init_map_x * 0.5) + Xstartx;
			yr = (drand() * init_map_y) - (init_map_y * 0.5) + Xstarty;

			x_rand.push_back(xr);
			x_rand.push_back(yr);

			// Nearest
			x_nearest = Nearest(V, x_rand);

			// Steer
			x_new = Steer(x_nearest, x_rand, eta);

			// ObstacleFree    1:free     -1:unkown (frontier region)      0:obstacle
			char checking = ObstacleFree(x_nearest, x_new, mapData);

			geometry_msgs::Point tmp;
			switch (checking)
			{
			case -1:
				exploration_goal.header.frame_id = mapData.header.frame_id;
				exploration_goal.point.x = x_new[0];
				exploration_goal.point.y = x_new[1];
				exploration_goal.point.z = 0.0;
				tmp.x = x_new[0];
				tmp.y = x_new[1];
				tmp.z = 0.0;
				points.points.push_back(tmp);
				point_pub.publish(exploration_goal);
				break;
			case 1:
				V.push_back(x_new);
				tmp.x = x_new[0];
				tmp.y = x_new[1];
				tmp.z = 0.0;
				lines.points.push_back(tmp);
				tmp.x = x_nearest[0];
				tmp.y = x_nearest[1];
				tmp.z = 0.0;
				lines.points.push_back(tmp);
				break;
			}

			// update publish tree's markers array
			marker_array.markers.push_back(lines);
			marker_array.markers.push_back(points);
			marker_pub.publish(marker_array);

			points.points.clear();
		}
	}
	return 0;
}
