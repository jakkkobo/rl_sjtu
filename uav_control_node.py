#!/usr/bin/env python

import rospy
import sys, getopt
import numpy as np
from scipy.optimize import minimize
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, PoseStamped, TwistStamped, PointStamped, Point, Vector3
from nav_msgs.msg import Path, Odometry
from gazebo_msgs.msg import ModelStates
from mavros_msgs.msg import PositionTarget

class ControlNode:
    def __init__(self, ID=0):

        # Initialize the global variable to None
        self.subscriber_message_uavpose = Point(0,0,0)
        self.subscriber_message_uavvel = Twist()
        self.control_uavvel = Twist()

        
        self.goal_position = Point(0,0,0)
        self.t_value = 0
        self.stop = False
        self.reset = True
        self.ID=ID

        # Initialize the ROS node
        rospy.init_node('Control_node', anonymous=True)

        # Create a publisher that runs at 50 Hz
        self.publisher_velocity = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        rospy.Timer(rospy.Duration(1.0/20), self.publish_velocity)

        # Create a subscriber
        rospy.Subscriber("uav_" + str(self.ID) + "/agent/actions", TwistStamped, self._subscriber_agent_callback)
        rospy.Subscriber("uav_" + str(self.ID) + "/agent/stop", Bool, self._subscriber_stop_callback)
        rospy.Subscriber("uav_" + str(self.ID) + "/agent/reset", Bool, self._subscriber_reset_callback)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._gazebo_pose_callback)

    def _reset_takeoff(self):
        goal = Point(0,0,1)
        # calculate velocity to reach the goal based on uav position
        # get the current position
        position = self.subscriber_message_uavpose
        # calculate the velocity
        velocity = Twist(Vector3(0,0,0), Vector3(0,0,0))
        #clip velocity to max 1 m/s
        velocity.linear.x = np.clip(goal.x - position.x, -1, 1)
        velocity.linear.y = np.clip(goal.y - position.y, -1, 1)
        velocity.linear.z = np.clip(goal.z - position.z, -1, 1)
        
        return velocity

    def _subscriber_reset_callback(self, msg:Bool):
        self.reset = msg.data
        print("Reset: ", self.reset)
            

    def publish_velocity(self, event):
        # Publish the message
        velocity = self.control_uavvel
        
        if self.stop == True:
            velocity = Twist(Vector3(0,0,0), Vector3(0,0,0))
    
        if self.reset == True:
            velocity = self._reset_takeoff()

        self.publisher_velocity.publish(velocity)


    def _subscriber_stop_callback(self, msg:Bool):
        self.stop = msg.data
        print("Stop: ", self.stop)
    
    def _gazebo_pose_callback(self, msg:ModelStates):
            # Get the list of model names
            model_name = msg.name
            name = 'sjtu_drone' #+ str(self.ID)
            #print(name)

            # Get the index of the 'sjtu_drone' model
            try:
                sjtu_drone_index = model_name.index(name)
            except ValueError:
                rospy.logwarn("%s model not found in ModelStates",name)
                return

            # Get the pose of the 'sjtu_drone' model
            sjtu_drone_pose = msg.pose[sjtu_drone_index]
            sjtu_drone_velocity = msg.twist[sjtu_drone_index]

            #check if velocity is nan
            if np.isnan(sjtu_drone_velocity.linear.x):
                sjtu_drone_velocity.linear.x = 0
            if np.isnan(sjtu_drone_velocity.linear.y):
                sjtu_drone_velocity.linear.y = 0
            if np.isnan(sjtu_drone_velocity.linear.z):
                sjtu_drone_velocity.linear.z = 0

            # Print the pose information
            self.subscriber_message_uavpose = sjtu_drone_pose.position
            self.subscriber_message_uavvel = sjtu_drone_velocity
            #rospy.loginfo("sjtu_drone Pose:\n{}".format(self.subscriber_message_uavpose))


    # subscribe the action from the agent
    def _subscriber_agent_callback(self, msg:TwistStamped):

        self.reset = False
        self.stop = False
        received_vel = msg.twist

        # sum up the linear and z angular velocity
        uavvel = Twist(Vector3(0,0,0), Vector3(0,0,0))
        

        if self.stop == False:
            uavvel.linear.x = received_vel.linear.x + self.subscriber_message_uavvel.linear.x
            uavvel.linear.y = received_vel.linear.y + self.subscriber_message_uavvel.linear.y
            uavvel.linear.z = received_vel.linear.z + self.subscriber_message_uavvel.linear.z
            # uavvel.angular.z = received_vel.angular.z + self.subscriber_message_uavvel.angular.z

        #limit the velocity to 1 m/s in x and y direction and 0.5 m/s in z direction and 0.174 rad/s in z direction
        uavvel = self.velocity_limiter(uavvel, 1, 1, 0.5, 0.0)
        

        # Publish the message
        self.publisher_velocity.publish(uavvel)

        self.control_uavvel = uavvel

    def velocity_limiter(self, uavvel, velx, vely, velz, angz):
        if uavvel.linear.x > velx:
            uavvel.linear.x = velx
        elif uavvel.linear.x < -velx:
            uavvel.linear.x = -velx
        if uavvel.linear.y > vely:
            uavvel.linear.y = vely
        elif uavvel.linear.y < -vely:
            uavvel.linear.y = -vely
        if uavvel.linear.z > velz:
            uavvel.linear.z = velz
        elif uavvel.linear.z < -velz:
            uavvel.linear.z = -velz
        # if uavvel.angular.z > angz:
        #     uavvel.angular.z = angz
        # elif uavvel.angular.z < -angz:
        #     uavvel.angular.z = -angz
        
        return uavvel

    def run(self):
        # Spin the node to receive and process messages
        rospy.spin()

def get_id(argv):
    ID=0
    opts, args = getopt.getopt(argv,":i:",["index="])
    for opt, arg in opts:
        # print(opt," ",arg)
        if opt in ("-i","--index"):
            ID = arg
    return ID

class ControlNode_PX4:
    def __init__(self, ID=0):

        # Initialize the global variable to None
        self.subscriber_message_uavpose = Point(0,0,0)
        self.subscriber_message_uavvel = Twist()
        self.control_uavvel = TwistStamped()

        
        self.goal_position = Point(0,0,0)
        self.t_value = 0
        self.stop = False
        self.reset = True
        self.ID=ID

        # Initialize the ROS node
        rospy.init_node('Control_node', anonymous=True)

        # Create a publisher that runs at 50 Hz
        self.publisher_velocity = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=10)

        rospy.Timer(rospy.Duration(1.0/20), self.publish_velocity)

        # Create a subscriber
        rospy.Subscriber("uav_" + str(self.ID) + "/agent/actions", TwistStamped, self._subscriber_agent_callback)
        rospy.Subscriber("uav_" + str(self.ID) + "/agent/stop", Bool, self._subscriber_stop_callback)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._gazebo_pose_callback)

    def publish_velocity(self, event):
        # Publish the message
        velocity = self.control_uavvel
        
        if self.stop == True:
            velocity.twist = Twist(Vector3(0,0,0), Vector3(0,0,0))

        self.control_uavvel.header.stamp = rospy.Time.now()
        self.publisher_velocity.publish(velocity)


    def _subscriber_stop_callback(self, msg:Bool):
        self.stop = msg.data
        print("Stop: ", self.stop)
    
    def _gazebo_pose_callback(self, msg:ModelStates):
            # Get the list of model names
            model_name = msg.name
            name = 'iris' + str(self.ID)
            #print(name)

            # Get the index of the 'iris' model
            try:
                iris_index = model_name.index(name)
            except ValueError:
                rospy.logwarn("%s model not found in ModelStates",name)
                return

            # Get the pose of the 'iris' model
            iris_pose = msg.pose[iris_index]
            iris_velocity = msg.twist[iris_index]

            #check if velocity is nan
            if np.isnan(iris_velocity.linear.x):
                iris_velocity.linear.x = 0
            if np.isnan(iris_velocity.linear.y):
                iris_velocity.linear.y = 0
            if np.isnan(iris_velocity.linear.z):
                iris_velocity.linear.z = 0

            # Print the pose information
            self.subscriber_message_uavpose = iris_pose.position
            self.subscriber_message_uavvel = iris_velocity
            #rospy.loginfo("iris Pose:\n{}".format(self.subscriber_message_uavpose))


    # subscribe the action from the agent
    def _subscriber_agent_callback(self, msg:TwistStamped):

        received_vel = msg.twist

        # sum up the linear and z angular velocity
        uavvel = Twist(Vector3(0,0,0), Vector3(0,0,0))
        

        if self.stop == False:
            uavvel.linear.x = received_vel.linear.x + self.subscriber_message_uavvel.linear.x
            uavvel.linear.y = received_vel.linear.y + self.subscriber_message_uavvel.linear.y
            uavvel.linear.z = received_vel.linear.z + self.subscriber_message_uavvel.linear.z
            # uavvel.angular.z = received_vel.angular.z + self.subscriber_message_uavvel.angular.z

        #limit the velocity to 1 m/s in x and y direction and 0.5 m/s in z direction and 0.174 rad/s in z direction
        uavvel = self.velocity_limiter(uavvel, 1, 1, 0.5, 0.0)
        

        # Publish the message
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.twist = uavvel
        self.publisher_velocity.publish(msg)

        self.control_uavvel.twist = uavvel

    def velocity_limiter(self, uavvel, velx, vely, velz, angz):
        if uavvel.linear.x > velx:
            uavvel.linear.x = velx
        elif uavvel.linear.x < -velx:
            uavvel.linear.x = -velx
        if uavvel.linear.y > vely:
            uavvel.linear.y = vely
        elif uavvel.linear.y < -vely:
            uavvel.linear.y = -vely
        if uavvel.linear.z > velz:
            uavvel.linear.z = velz
        elif uavvel.linear.z < -velz:
            uavvel.linear.z = -velz
        # if uavvel.angular.z > angz:
        #     uavvel.angular.z = angz
        # elif uavvel.angular.z < -angz:
        #     uavvel.angular.z = -angz
        
        return uavvel

    def run(self):
        # Spin the node to receive and process messages
        rospy.spin()

class ControlNode_Sentinel:
    def __init__(self, ID=0):

        # Initialize the global variable to None
        self.subscriber_message_uavpose = PoseStamped()
        self.subscriber_message_uavvel = PointStamped()
        self.control_uavvel = PositionTarget()
        self.control_uavvel.header.frame_id = ""
        self.control_uavvel.coordinate_frame = 1
        self.control_uavvel.type_mask = 4039

        
        self.goal_position = Point(0,0,0)
        self.t_value = 0
        self.stop = False
        self.reset = True
        self.ID=ID

        # Initialize the ROS node
        rospy.init_node('Control_node', anonymous=True)

        # Create a publisher that runs at 50 Hz
        self.publisher_velocity = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)

        rospy.Timer(rospy.Duration(1.0/20), self.publish_velocity)

        # Create a subscriber
        rospy.Subscriber("uav_" + str(self.ID) + "/agent/actions", TwistStamped, self._subscriber_agent_callback)
        rospy.Subscriber("uav_" + str(self.ID) + "/agent/stop", Bool, self._subscriber_stop_callback)
        rospy.Subscriber("uav_" + str(self.ID) + "/agent/takeoff", Bool, self._subscriber_takeoff_callback)
        # rospy.Subscriber("/gazebo/model_states", ModelStates, self._gazebo_pose_callback)

        #Get pose and vel from drone
        rospy.Subscriber("/drone/local_wrt_fixed/pose", PoseStamped, self._mavros_pose_callback)
        rospy.Subscriber("/drone/local_wrt_fixed/velocity", PointStamped, self._mavros_vel_callback)


    def _subscriber_takeoff_callback(self, msg:Bool):
        # publish velocity to pose (0,0,1)
        if msg.data == True:
            while -self.subscriber_message_uavpose.pose.position.z > -1.0:
                self.control_uavvel.velocity.x = 0
                self.control_uavvel.velocity.y = 0
                self.control_uavvel.velocity.z = 0.5
                self.control_uavvel.header.stamp = rospy.Time.now()
                self.publisher_velocity.publish(self.control_uavvel)


    def publish_velocity(self, event):
        # Publish the message
        velocity = self.control_uavvel
        
        if self.stop == True:
            velocity.velocity.x = 0
            velocity.velocity.y = 0
            velocity.velocity.z = 0

        velocity.header.stamp = rospy.Time.now()
        self.publisher_velocity.publish(velocity)


    def _subscriber_stop_callback(self, msg:Bool):
        self.stop = msg.data
        print("Stop: ", self.stop)
    
    
    def _mavros_pose_callback(self, msg:PoseStamped):
        
        # Print the pose information
        self.subscriber_message_uavpose = msg


    def _mavros_vel_callback(self, msg:PointStamped):
        # Get the pose of the 'iris' model
        iris_velocity = msg.point
        
        # Print the pose information
        self.subscriber_message_uavvel = iris_velocity



    # subscribe the action from the agent
    def _subscriber_agent_callback(self, msg:TwistStamped):

        received_vel = msg.twist

        # sum up the linear and z angular velocity
        uavvel = Twist(Vector3(0,0,0), Vector3(0,0,0))
        

        if self.stop == False:
            uavvel.linear.x = received_vel.linear.x + self.subscriber_message_uavvel.x
            uavvel.linear.y = received_vel.linear.y + self.subscriber_message_uavvel.y
            uavvel.linear.z = received_vel.linear.z + self.subscriber_message_uavvel.z
            # uavvel.angular.z = received_vel.angular.z + self.subscriber_message_uavvel.angular.z

        #limit the velocity to 1 m/s in x and y direction and 0.5 m/s in z direction and 0.174 rad/s in z direction
        uavvel = self.velocity_limiter(uavvel, 1, 1, 0.5, 0.0)
        

        # Publish the message
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.coordinate_frame = 1
        msg.type_mask = 4039
        msg.velocity.x = -uavvel.linear.y
        msg.velocity.y = uavvel.linear.x
        msg.velocity.z = uavvel.linear.z
        self.publisher_velocity.publish(msg)

        self.control_uavvel = msg

    def velocity_limiter(self, uavvel, velx, vely, velz, angz):
        if uavvel.linear.x > velx:
            uavvel.linear.x = velx
        elif uavvel.linear.x < -velx:
            uavvel.linear.x = -velx
        if uavvel.linear.y > vely:
            uavvel.linear.y = vely
        elif uavvel.linear.y < -vely:
            uavvel.linear.y = -vely
        if uavvel.linear.z > velz:
            uavvel.linear.z = velz
        elif uavvel.linear.z < -velz:
            uavvel.linear.z = -velz
        # if uavvel.angular.z > angz:
        #     uavvel.angular.z = angz
        # elif uavvel.angular.z < -angz:
        #     uavvel.angular.z = -angz
        
        return uavvel

    def run(self):
        # Spin the node to receive and process messages
        rospy.spin()



if __name__ == '__main__':
    id=get_id(sys.argv[1:])
    print("ID: ",id)
    try:
        node = ControlNode_Sentinel(ID=id)
        node.run()
    except rospy.ROSInterruptException:
        pass
