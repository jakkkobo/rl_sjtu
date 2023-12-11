import time

import gym
import numpy as np
import rospy, math

from std_msgs.msg import Bool
from std_msgs.msg import Empty as empty_msg
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, PointStamped, Point
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import SetMode, CommandBool
from visualization_msgs.msg import Marker

from gazebo_msgs.msg import ModelStates

from gazebo_msgs.srv import DeleteModel, SpawnModel


from gym import spaces
from std_srvs.srv import Empty, EmptyRequest
from turtlesim.msg import Pose
import tf


def lmap(v, x, y) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


class UAVEnv(gym.Env):
    def __init__(
        self,
        index = 0,  # env index
        rate = 10, # simulation refresh rate
        steps_thr = 1024*2, 
    ):
        super().__init__()
        self.index = index

        #rospy.init_node("UAVEnv" + str(index))

        self.rate = rospy.Rate(rate)
        self._create_ros_pub_sub_srv()
        

        
        # Action dimension and boundaries
        self.act_dim = 3 # [vx, vy, vz]
        self.action_space = spaces.Box(
            low=-np.ones(self.act_dim), high=np.ones(self.act_dim), dtype=np.float32
        )
        self.act_bnd = {"vx": (-0.1, 0.1), "vy": (-0.1, 0.1), "vz":(-0.1, 0.1)} #, "wz":(-0.175, 0.175)} # What are the boudaries for the velocity and angular velocity

        self.obs_dim = 10  # [dist, UAVPose(3), WaypointPose(3), velocity(3)]
        self.observation_space = spaces.Box(
            low=-np.ones(self.obs_dim), high=np.ones(self.obs_dim), dtype=np.float32
        )
        self.obs_bnd = {"xyz": 10.}  # cubic space between (-100 to 100)

        self.rew_w = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )  # [dist, uav_x, uav_y, uav_z, g_x, g_z, g_y, vx, vy, vz]
        self.dist_threshold = 0.25  # done if dist < dist_threshold, final goal radius distance

        self.goal_name = "GoalPoint"
        self.uav_pose = np.zeros(3)
        self.goal = np.zeros(3)
        self.prev_actions = np.zeros(3)
        self.uav_velocity = np.zeros(3)
        

        self.steps = 0
        self.total_ts = 0
        self.steps_thr = steps_thr
        self.prev_dist = 0
        self.got_goal = 1
        

        self.reset()
        rospy.loginfo("[ENV Node " + str(self.index) + " ] Initialized")

    def step(self, action):
        if not rospy.is_shutdown():
        
            self.steps += 1
            self.total_ts += 1

            #self.publish_action(np.zeros(6))
            self.publish_action(action) # send the action to the Bezier node
            observation = self.observe() # get the observation
            reward = self.compute_reward(observation) # compute the reward
            done = self.is_terminal(observation) # check if the episode is done

            # Check if the UAV is out of the boundaries
            #if self.steps >= 50:
            if not(-22.0 <= self.uav_pose[0] <= 22 and -22 <= self.uav_pose[1] <= 22 and -22 <= self.uav_pose[2] <= 22):
                done = True
                reward = -self.steps_thr
                print(reward)
            
            info = {}
            self.rate.sleep()
            return observation, reward, done, info
        else:
            rospy.logerr("rospy is shutdown")

    def reset(self): # reset the training  - send the vehicle to init postion ToDo
        self.steps = 0
        
        self.stop_publisher.publish(Bool(True))  # TODO ADD Action to stop the UAV
        # self._reset_bezier("stop")
        print("Reseting")
        time.sleep(5)

        # self.reset_publisher.publish(Bool(True))
        # print("Pause Sim")
        # self.pause_sim()

        time.sleep(1)

        print("Reset World")
        self.reset_world()


        time.sleep(2)

        if self.got_goal ==1:
            try:
                self._clear_goal()
            except:
                None
            self.set_goal()
            self.got_goal=0
        

        # self.stop_publisher.publish(Bool(False))
        # print("Unpause Sim")
        # self.unpause_sim()
        # self.reset_publisher.publish(Bool(True))

        self.takeoff_publisher.publish(empty_msg())

        time.sleep(2)
        print("New Goal =",self.goal)

        return self.observe()
    

    def publish_action(self, action, type="default"): # action changed
        action = self._proc_action(action)

        #check if action value are nan
        if np.isnan(action[0]):
            action[0] = 0
        if np.isnan(action[1]):
            action[1] = 0
        if np.isnan(action[2]):
            action[2] = 0
        # if np.isnan(action[3]):
        #     action[3] = 0

        sender_action = TwistStamped()
        sender_action.header.stamp = rospy.Time.now()
        sender_action.twist.linear.x = action[0]
        sender_action.twist.linear.y = action[1]
        sender_action.twist.linear.z = action[2]
        # sender_action.twist.angular.z = action[3]

        self.prev_actions = action
        self.prev_dist = np.linalg.norm(self.uav_pose - self.goal, 2)

        self.agent_actions_publisher.publish(sender_action)
        

    def observe(self, n_std=0.1):
        relative_dist = np.linalg.norm(self.uav_pose - self.goal, 2)
        

        return np.concatenate([np.array([relative_dist]), self.uav_pose, self.goal, self.uav_velocity]) + np.random.normal(
            0, n_std, self.obs_dim
        )

    def compute_reward(self, obs):

        ideal_velocity_vector = self.publish_ideal_velocity(self.uav_pose, self.goal)
        velocity_vector = self.publish_current_velocity(self.uav_pose, self.uav_velocity)


        # Calculate the angular error between the ideal velocity and the current velocity
        angular_error = np.arccos(np.clip(np.dot(ideal_velocity_vector, velocity_vector), -1.0, 1.0))
        print("Angular Error: " + str(angular_error))

        velocity_reward = np.clip(-np.abs(angular_error)/np.pi, -1, 0) # the min reward is -1 
        print("Vel: " + str(velocity_reward))

        # Calculate distance reward
        distance_reward = np.clip(np.dot(self.rew_w, -np.abs(obs)/76), -1, 0) #76 is the max diagonal distance of the bounding box, therefore the max distance reward is -1
        # print("Dist: " + str(distance_reward))
        
        reward = (distance_reward + velocity_reward)
        print("Total Reward:" + str(reward))


        #check if reward is nan
        if np.isnan(reward):
            reward = prev_reward
        prev_reward = reward
        return reward
    
    
    def publish_current_velocity(self, uav_pose, uav_velocity):

        # normalize vector
        vector_norm = np.linalg.norm(uav_velocity, 2)
        vector_norm = uav_velocity/vector_norm

        #Create the arrow marker for visualization
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "map"
        arrow_marker.header.stamp = rospy.Time.now()
        arrow_marker.ns = "uav_velocity_vector"
        arrow_marker.id = 0
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        arrow_marker.pose.orientation.w = 1.0
        arrow_marker.scale.x = 0.1
        arrow_marker.scale.y = 0.2
        arrow_marker.scale.z = 0.2
        arrow_marker.color.r = 1.0
        arrow_marker.color.g = 0.0
        arrow_marker.color.b = 0.0
        arrow_marker.color.a = 1.0
        arrow_marker.lifetime = rospy.Duration(0.1)
        arrow_marker.points.append(Point(uav_pose[0], uav_pose[1], uav_pose[2]))
        arrow_marker.points.append(Point(uav_pose[0] + vector_norm[0], uav_pose[1] + vector_norm[1], uav_pose[2] + vector_norm[2]))
        self.velocity_vector_publisher.publish(arrow_marker)

        return vector_norm


    def publish_ideal_velocity(self, uav_pose, goal):

        # get the vector
        vector = goal - uav_pose

        # normalize vector
        vector_norm = np.linalg.norm(vector, 2)
        vector_norm = vector/vector_norm

        #Create the arrow marker for visualization
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "map"
        arrow_marker.header.stamp = rospy.Time.now()
        arrow_marker.ns = "uav_goal_vector"
        arrow_marker.id = 0
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        arrow_marker.pose.orientation.w = 1.0
        arrow_marker.scale.x = 0.1
        arrow_marker.scale.y = 0.2
        arrow_marker.scale.z = 0.2
        arrow_marker.color.r = 0.0
        arrow_marker.color.g = 1.0
        arrow_marker.color.b = 0.0
        arrow_marker.color.a = 1.0
        arrow_marker.lifetime = rospy.Duration(0.1)
        arrow_marker.points.append(Point(uav_pose[0], uav_pose[1], uav_pose[2]))
        arrow_marker.points.append(Point(uav_pose[0] + vector_norm[0], uav_pose[1] + vector_norm[1], uav_pose[2] + vector_norm[2]))
        self.ideal_velocity_vector_publisher.publish(arrow_marker)

        return vector_norm


    def set_goal(self):
        #self.goal = self._random_goal()
        self.goal = np.array([8,8,8])

        #publish goal point
        goal_msg = PointStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"
        goal_msg.point.x = self.goal[0]
        goal_msg.point.y = self.goal[1]
        goal_msg.point.z = self.goal[2]
        self.goal_publisher.publish(goal_msg)

    def is_terminal(self, observation):
        done = observation[0] <= self.dist_threshold or self.steps_thr - self.steps <=0 #or (self.uav_pose[2]<3 and self.steps >500)
        #print(self.steps, self.total_ts)

        if observation[0] <= self.dist_threshold:
            self.got_goal=1
        return done

    def _create_ros_pub_sub_srv(self):


        # Agent publishers
        self.agent_actions_publisher = rospy.Publisher("uav_" + str(self.index) + "/agent/actions", TwistStamped, queue_size=1)
        self.stop_publisher = rospy.Publisher("uav_" + str(self.index) + "/agent/stop", Bool, queue_size=1)
        self.reset_publisher = rospy.Publisher("uav_" + str(self.index) + "/agent/reset", Bool, queue_size=1)

        #UAV control
        self.takeoff_publisher = rospy.Publisher("drone/takeoff", empty_msg, queue_size=1)

        #Get the current UAV postion from Gazebo
        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self._gazebo_pose_callback
        )

        # Visualization only
        self.velocity_vector_publisher = rospy.Publisher("uav_" + str(self.index) + "/agent/velocity_vector", Marker, queue_size=1)
        self.ideal_velocity_vector_publisher = rospy.Publisher("uav_" + str(self.index) + "/agent/ideal_velocity_vector", Marker, queue_size=1)
        self.goal_publisher = rospy.Publisher("uav_" + str(self.index) + "/agent/goal", PointStamped, queue_size=1)
        self.uav_pose_publisher = rospy.Publisher("uav_" + str(self.index) + "/agent/pose", PointStamped, queue_size=1)

        #Gazebo Services for pause and return simulation
        self.delete = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        self.spawn = rospy.ServiceProxy("gazebo/spawn_urdf_model", SpawnModel)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy(
            "/gazebo/reset_simulation", Empty
        )
        self.reset_world_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        time.sleep(0.1)

    # Mavros Subscriber deactivated !!! ----------
    #def _mavros_pose_callback(self, msg: PoseStamped): 
    #    self.uav_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    #-----------------------------------------    

    def publish_uav_pose(self, uav_pose):

        uav_pose_msg = PointStamped()
        uav_pose_msg.header.stamp = rospy.Time.now()
        uav_pose_msg.header.frame_id = "map"
        uav_pose_msg.point.x = uav_pose[0]
        uav_pose_msg.point.y = uav_pose[1]
        uav_pose_msg.point.z = uav_pose[2]
        self.uav_pose_publisher.publish(uav_pose_msg)

    def _gazebo_pose_callback(self, msg:ModelStates):
        # Get the list of model names
        model_name = msg.name

        # Get the index of the 'sjtu_drone' model
        try:
            sjtu_drone_index = model_name.index("sjtu_drone")
        except ValueError:
            rospy.logwarn("'sjtu_drone' model not found in ModelStates")
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
        self.uav_pose = np.array([sjtu_drone_pose.position.x, sjtu_drone_pose.position.y, sjtu_drone_pose.position.z])
        self.uav_velocity = np.array([sjtu_drone_velocity.linear.x, sjtu_drone_velocity.linear.y, sjtu_drone_velocity.linear.z]) #, sjtu_drone_velocity.angular.z])
        #rospy.loginfo("sjtu_drone Pose:\n{}".format(self.uav_pose))

        # Publish the UAV pose
        self.publish_uav_pose(self.uav_pose)

    def _proc_action(self, action, noise_std=0.3): # generates action between -1 and 1 and then scale up to the boundaries
        proc = action + np.random.normal(0, noise_std, action.shape)
        proc = np.clip(proc, -1, 1)
        
        # scale the action to the boundaries
        proc[0] = lmap(proc[0], [-1, 1], self.act_bnd["vx"])
        proc[1] = lmap(proc[1], [-1, 1], self.act_bnd["vx"])
        proc[2] = lmap(proc[2], [-1, 1], self.act_bnd["vz"])
        # proc[3] = lmap(proc[3], [-1, 1], self.act_bnd["wz"])

        return proc

    def _random_goal(self):
        goal = np.random.uniform(-1, 1, 3)
        goal[0:2] = lmap(goal[0:2], [-1, 1], self.act_bnd["XY"])
        goal[2] = lmap(goal[2], [-1, 1], self.act_bnd["Z"])
        #goal = self.obs_bnd["xyz"]*(goal)
        return goal

    def _clear_goal(self):
        self.goal=np.zeros(3)
        #kill_obj = KillRequest()
        #kill_obj.name = self.goal_name
        #self.kill_srv(kill_obj)


    def reset_world(self):
        """reset gazebo world"""
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as err:
            print("/gazebo/reset_world service call failed", err)

        rospy.logdebug("Finich Reset")

    def pause_sim(self):
        """pause simulation with ros service call"""
        rospy.logdebug("PAUSING START")
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as err:
            print("/gazebo/pause_physics service call failed", err)

        rospy.logdebug("PAUSING FINISH")

    def unpause_sim(self):
        """unpause simulation with ros service call"""
        rospy.logdebug("UNPAUSING START")
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as err:
            print("/gazebo/unpause_physics service call failed", err)

        rospy.logdebug("UNPAUSING FiNISH")

    def set_uav_mode(self,new_mode):
        mode = SetMode()
        mode.custom_mode = new_mode
        """reset gazebo world"""
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            set_mode_response = self.set_mode(0, mode.custom_mode)  # 0 is the base mode
            if set_mode_response.mode_sent:
                rospy.loginfo("Changed to %s mode",new_mode)
            else:
                rospy.logwarn("Failed to change to %s mode",new_mode)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))
        


    @classmethod
    def compute_angle(cls, goal_pos: np.array, obs_pos: np.array) -> float:
        pos_diff = obs_pos - goal_pos
        goal_yaw = np.arctan2(pos_diff[1], pos_diff[0]) - np.pi
        ang_diff = goal_yaw - obs_pos[2]

        if ang_diff > np.pi:
            ang_diff -= 2 * np.pi
        elif ang_diff < -np.pi:
            ang_diff += 2 * np.pi

        return ang_diff

    def render(self):
        raise NotImplementedError

    def close(self):
        rospy.signal_shutdown("Training Complete") 


    def _goal_pose_callback(self, msg: Pose):
        self.goal = np.array([msg.x, msg.y, msg.theta])


if __name__ == "__main__":
    rospy.init_node("UAVEnv")

    env = UAVEnv()
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        action = np.ones_like(action)  # [thrust, ang_vel]
        obs, reward, terminal, info = env.step(action)
        
    env.reset()
    print("Finish test")
