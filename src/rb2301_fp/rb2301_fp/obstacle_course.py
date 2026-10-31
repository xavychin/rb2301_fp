import argparse
import configparser
import math
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.qos import ReliabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

np.set_printoptions(2, suppress=True, threshold=np.inf) # Print numpy arrays to specified d.p., suppress scientific notation and do not truncate
set_logger_level("obstaclecourse", level=LoggingSeverity.INFO) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG  


_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))

def wrap_deg(angle:float) -> float:
    '''Wrap an angle in degrees into [-180, 180).'''
    return (angle + 180.0) % 360.0 - 180.0

def optitrack_to_maze(x:float, y:float, heading_deg:float, frame:dict) -> tuple:
    '''Optitrack-frame pose (x, y, heading in degrees) -> CA2 maze-frame pose (see optitrack_variables.config).'''
    dx, dy = x - frame["origin_x"], y - frame["origin_y"]
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - frame["rotation_deg"] - frame["heading_offset_deg"]))

def maze_to_course(x:float, y:float, heading_deg:float, course:dict) -> tuple:
    '''CA2 maze-frame pose -> final-project course frame (origin/rotation of the course in the maze frame, [course] in the config).'''
    dx, dy = x - course["origin_x"], y - course["origin_y"]
    r = math.radians(course["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy, wrap_deg(heading_deg - course["rotation_deg"]))

def load_frame(robot_number:int) -> tuple:
    '''Read optitrack_variables.config: returns (frame, course) for this robot.'''
    parser = configparser.ConfigParser()
    path = os.path.join(_PACKAGE_DIR, "optitrack_variables.config")
    if not parser.read(path):
        raise FileNotFoundError(f"Could not read {path}")
    frame = {k: parser.getfloat("frame", k) for k in ("origin_x", "origin_y", "rotation_deg")}
    raw = parser.get("frame", "heading_offset_deg", fallback="auto").strip().lower()
    # Rigid bodies are created facing +y of the maze frame, so Motive's zero heading is maze +y (90 deg)
    frame["heading_offset_deg"] = wrap_deg(-(frame["rotation_deg"] + 90.0)) if raw == "auto" else float(raw)
    if parser.has_section("heading_offsets") and parser.has_option("heading_offsets", str(robot_number)):
        frame["heading_offset_deg"] = parser.getfloat("heading_offsets", str(robot_number))
    course = {k: parser.getfloat("course", k, fallback=0.0) for k in ("origin_x", "origin_y", "rotation_deg")}
    return frame, course

class LidarNode(Node):
    def __init__(self):
        """Node constructor"""
        super().__init__("LidarNode")
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to lidar
        self.is_simulation = None

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 360: 
            self.is_simulation = True
        else:
            self.is_simulation = False

class ObstacleCourseNode(Node):
    '''Node to navigate obstacle course, using pose from either gazebo odometer or optitrack and lidar scan data'''
    def __init__(self, is_simulation:bool=True, calibrate:bool=False):
        super().__init__('obstaclecourse')
        self.get_logger().info("Starting ObstacleCourseNode")

        self.is_simulation = is_simulation
        self.calibrate = calibrate # Real robot: only print poses, never drive
        self.raw_pose = None # Pose exactly as Optitrack reported it (real robot only)
        self.frame = self.course = None
        self._ticks = 0

        if is_simulation:
            self.max_translate_velocity = 1.4
        else:
            self.max_translate_velocity = 0.3 # Please keep this in place; 0.3m/s is more than fast enough 

        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to LiDAR scan data

        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, 2)
        else:
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            # On the Bingda robots ROS_DOMAIN_ID is the robot's number, so the Optitrack topic is picked automatically
            domain = os.environ.get("ROS_DOMAIN_ID", "")
            self.robot_number = int(domain) if domain.isdigit() and 1 <= int(domain) <= 99 else 3
            self.frame, self.course = load_frame(self.robot_number)
            self.get_logger().info(f"Reading pose from /vrpn_mocap/bingda_{self.robot_number:03d}/pose")

            self.map_sub = self.create_subscription( 
                PoseStamped,
                f'/vrpn_mocap/bingda_{self.robot_number:03d}/pose',
                self.optitrack_callback, 
                qos_profile
                )
            
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node       
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.pose = None

        ##Variables
        self.prev_dir = "front"   # Travel direction: "front", "left", "rear" or "right"
        self.dir = 0.2            # Strafe side when blocked: > 0 is left of travel, < 0 is right of travel
        self.turning = False      # True while dodging an obstacle ahead
        self.depth = 0.0          # How far ahead the lane must be free to finish the dodge
        self.backing = False      # True once a dodge has had to back away
        self.leg = 0              # Which leg of the course the robot is on (see legs below)
        self.wait = 0             # Ticks spent waiting for a moving obstacle to pass
        self.prev_scan = None     # Scan the moving obstacles were last measured in
        self.scan_ticks = 0       # Ticks since the scan last changed
        self.scans = 0            # Scans seen since the moving obstacles started being tracked
        self.paths = []           # What is known about each moving obstacle's path (see track_moving)
        self.leg_start = None     # Pose where the current leg started
        self.crossing = None      # Progress at which the robot is past the moving obstacle's path it is crossing
        self.last_cmd = np.zeros(2)  # Last (forward, sideways) command sent
        self.prev_pose = None     # Pose on the last tick
        self.cross_ratio = 0.5    # Speed the robot really crosses moving obstacles' paths at, per unit commanded
        self.last_scan = None

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 360:
            self.last_scan = np.array(msg.ranges)
        else:
            self.last_scan = np.array(msg.ranges)[::2] 

    def yaw_from_quaternion(self, q):
        '''Returns yaw angle (in rad) for orientation based on given quaternion input q'''
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def optitrack_callback(self, msg:PoseStamped):
        '''Callback to calculate 2D pose info from Optitrack node. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        x, y = msg.pose.position.x, msg.pose.position.y
        heading = np.rad2deg(self.yaw_from_quaternion(msg.pose.orientation))
        self.raw_pose = np.array((x,y,heading))
        x, y, heading = optitrack_to_maze(x, y, heading, self.frame) # CA2 maze frame, see optitrack_variables.config
        self.pose = np.array((x,y,heading))
        return self.pose

    def to_course(self, pose=None):
        '''Convert a pose [x, y, heading] (default: self.pose) from the CA2 maze frame to the final-project course frame. Optional; see [course] in optitrack_variables.config'''
        pose = self.pose if pose is None else pose
        return np.array(maze_to_course(pose[0], pose[1], pose[2], self.course)) if self.course else np.array(pose)

    def odometer_callback(self, msg):
        '''Callback to calculate 2D pose info from Gazebo odomoter. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        latest_pose_msg = msg.pose.pose 
        heading = np.rad2deg(self.yaw_from_quaternion(latest_pose_msg.orientation))
        self.pose = np.array((latest_pose_msg.position.x, latest_pose_msg.position.y, heading))
        return self.pose

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        '''Publishes a Twist message to ROS to move a robot. Inputs are x and y linear velocities, as well as turn (z-axis yaw) angular velocity.'''
        twist_msg = Twist()
        x = np.clip(x, -self.max_translate_velocity, self.max_translate_velocity)
        y = np.clip(y, -self.max_translate_velocity, self.max_translate_velocity)
        turn = np.clip(turn, -self.max_translate_velocity*2, self.max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)

    def timer_callback(self):
        """Controller loop. Insert path planning and PID control logic here"""
        if self.pose is None:
            print("No pose detected")
            return # Does not run if no pose received from Odom or Optitrack

        if self.calibrate and self.raw_pose is not None:
            self._ticks += 1
            if self._ticks % 10 == 0: # twice a second
                ox, oy, oh = self.raw_pose
                self.get_logger().info(f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)  ->  maze ({self.pose[0]:+.3f}, {self.pose[1]:+.3f}, {self.pose[2]:+7.1f} deg)")
            return # Calibration mode: look, don't drive

        ###### INSERT CODE HERE ######
        self.get_logger().info(f"Pose: {self.pose}")

        if self.last_scan is None:
            return # No LiDAR scan received yet
        scan = np.where(self.last_scan > 0.05, self.last_scan, np.inf) # Readings this close are invalid

        # Minimum distance in each direction.
        sectors = {
            "front":       list(range(338, 360)) + list(range(0, 23)),
            "front_left":  range(23, 68),
            "left":        range(68, 113),
            "rear_left":   range(113, 158),
            "rear":        range(158, 203),
            "rear_right":  range(203, 248),
            "right":       range(248, 293),
            "front_right": range(293, 338),
        }

        distance = {
            name: np.min(np.take(scan, indices, mode="wrap"))
            for name, indices in sectors.items()
        }

        # Priority order: direction, forward speed, sideways speed.
        rules = [
            ("front",  0.2,  0.0),
            ("right",  0.0, -0.2),
            ("left",   0.0,  0.2),
            ("rear",  -0.2,  0.0),
        ]

        velocity = {direction: np.array((forward, sideways)) for direction, forward, sideways in rules}
        BODY = ((-0.15, -0.11), (-0.15, 0.11), (0.07, -0.11), (0.07, 0.11)) # Robot corners from the LiDAR (x forward, y left)
        MARGIN = 0.03      # Clearance kept around the robot when choosing a lane
        HOLD = 0.015       # Smaller clearance for staying in a lane already chosen
        SAFE = 0.02        # Anything closer than this to the robot pushes it away
        STOP = 0.1         # Distance ahead of the robot at which the way counts as blocked
        LOOK = 0.5         # How far ahead a lane must be free to strafe to it
        CORRIDOR = 1.2     # Side clearances adding up to less than this mean the robot is in a corridor
        WALL_DEPTH = 0.6   # A leg ends when its end wall is this close (half a corridor width)
        WALL_WIDTH = 0.8   # Width of a flat run across the lane that makes it a wall, not an obstacle
        GATE = 0.3         # Gate walls must run sideways from closer than this on both sides
        WAIT = 200         # Ticks (10 s) to wait for a fully safe crossing before settling for a near one
        ZONE = 1.0         # How far ahead of the robot moving obstacles are tracked
        WAIT_LINE = 0.06   # Wait this far short of a moving obstacle's path
        MIN_SPEED = 0.2    # Slower commands barely move the robot (in Gazebo at least)
        MOVING_SIZE = (0.2, 0.4) # Sideways length of a moving obstacle (they are 0.3 m)

        # Moving obstacles' paths are crossed at full speed
        cross_speed = self.max_translate_velocity / 0.2 # Times the normal speed

        # How fast the robot really moves at full speed (0.05 s per tick), measured
        # from the pose: it can be well below what it is told (it is in Gazebo)
        commanded = np.linalg.norm(self.last_cmd)
        if self.prev_pose is not None and commanded >= 0.99 * self.max_translate_velocity:
            ratio = np.linalg.norm(self.pose[:2] - self.prev_pose) / (commanded * 0.05)
            self.cross_ratio = 0.95 * self.cross_ratio + 0.05 * np.clip(ratio, 0.1, 1.5)
        self.prev_pose = self.pose[:2].copy()

        # The course: up the start corridor, right along the side corridor
        # into the end room, forward through the gate gap, then forward past
        # the moving obstacles to the end wall. Only the walls are fixed, so
        # each leg ends on what the LiDAR sees, not on a position.
        legs = ["front", "right", "front", "front"]
        heading = {"front": 0, "left": 90, "rear": 180, "right": 270}

        def project(direction):
            '''Scan points as (distance along direction, distance across it, positive to its left).'''
            angle = np.deg2rad(np.arange(len(scan)) * 360.0 / len(scan) - heading[direction])
            with np.errstate(invalid="ignore"):
                return scan * np.cos(angle), scan * np.sin(angle)

        def body(direction):
            '''Robot extent (along min, along max, across min, across max) seen from this direction.'''
            h = np.deg2rad(heading[direction])
            along = [x * np.cos(h) + y * np.sin(h) for x, y in BODY]
            across = [-x * np.sin(h) + y * np.cos(h) for x, y in BODY]
            return min(along), max(along), min(across), max(across)

        def wall_ahead(direction, depth):
            '''True if something flat, closer than depth, runs across the robot's lane and
            at least WALL_WIDTH wide (it may end at a side wall). Obstacles are at most
            0.5 m wide and two side by side are rarely level, so only a wall does this.'''
            along, across = project(direction)
            near = (along > 0) & (along < depth)
            edges = np.arange(-1.2, 1.2, 0.05)
            nearest = np.array([along[near & (across >= b) & (across < b + 0.05)].min()
                                if np.any(near & (across >= b) & (across < b + 0.05)) else np.inf for b in edges])
            lane = np.abs(edges + 0.025) < 0.1
            if not np.all(np.isfinite(nearest[lane])):
                return False
            level = np.abs(nearest - np.median(nearest[lane])) < 0.1
            # Grow the run of level bins outwards from the lane
            centre = np.argmin(np.abs(edges + 0.025))
            lo = hi = centre
            while lo > 0 and level[lo - 1]:
                lo -= 1
            while hi < len(edges) - 1 and level[hi + 1]:
                hi += 1
            return (hi - lo + 1) * 0.05 >= WALL_WIDTH

        def lane_blocked(direction, depth, offset=0.0, margin=MARGIN):
            '''True if anything is in the robot-wide lane up to depth beyond the robot's front.'''
            along, across = project(direction)
            back, front, lo, hi = body(direction)
            return np.any((along > front - 0.02) & (along < front + depth) &
                          (across > lo + offset - margin) & (across < hi + offset + margin))

        def beside_blocked(direction, offset):
            '''True if anything beside the robot is in the way of strafing by offset.'''
            along, across = project(direction)
            back, front, lo, hi = body(direction)
            return np.any((along > back - HOLD) & (along < front + HOLD) &
                          (across > min(lo, lo + offset) - HOLD) & (across < max(hi, hi + offset) + HOLD))

        def slack(direction, depth):
            '''How far (up to 0.1 m) the robot could move (left, right) and keep its lane free.'''
            room = []
            for sign in (1, -1):
                steps = [o for o in np.arange(0.01, 0.105, 0.01)
                         if lane_blocked(direction, depth, o * sign, HOLD) or beside_blocked(direction, o * sign)]
                room.append(min(steps) - 0.01 if steps else 0.1)
            return room

        def find_lane(direction):
            '''Sideways offsets (positive = left of travel) to the nearest free lane on each
            side, found without strafing into anything beside the robot. None if no lane.'''
            lanes = {}
            for sign in (1, -1):
                for offset in np.arange(0.01, 1.5, 0.01) * sign:
                    if beside_blocked(direction, offset):
                        break # Something beside the robot: can't strafe further this way
                    if not lane_blocked(direction, self.depth, offset):
                        lanes[sign] = offset
                        break
            return lanes

        def push_away():
            '''(forward, left) push away from anything within SAFE of the robot, or None.'''
            x, y = project("front")
            (x0, y0), (x1, y1) = BODY[0], BODY[3]
            inside = (x > x0 - SAFE) & (x < x1 + SAFE) & (y > y0 - SAFE) & (y < y1 + SAFE)
            if not np.any(inside):
                return None
            # Push away from whichever face of the robot each point is nearest
            gaps = np.stack((x[inside] - x0, x1 - x[inside], y[inside] - y0, y1 - y[inside]))
            faces = np.argmin(gaps, axis=0)
            push = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])[faces].sum(axis=0)
            return push / max(np.linalg.norm(push), 1e-9)

        def in_gate():
            '''True if a wall runs sideways away from the robot on both sides, level with
            it: the gate. Obstacles beside the robot are round or short, so they don't.'''
            along, across = project("front")
            level = np.abs(along) < 0.1
            sides = []
            for sign in (1, -1):
                a = across[level] * sign
                a = a[(a > 0) & (a < 0.8)]
                sides.append(len(a) > 0 and a.min() < GATE and a.max() - a.min() > 0.2)
            return all(sides)

        def progress(direction):
            '''Distance travelled in direction since the leg started (from the pose).'''
            h = np.deg2rad(self.pose[2] + heading[direction])
            return float(np.dot(self.pose[:2] - self.leg_start, (np.cos(h), np.sin(h))))

        def track_moving(direction):
            '''Track the moving obstacles ahead. They move sideways along fixed paths, so
            each is seen as a 0.3 m sideways line (walls are longer, or run the other
            way). For each path remember how far along it is, the obstacle's position,
            speed and direction, and the furthest points it has been seen to turn
            round at.'''
            self.scan_ticks += 1
            if self.last_scan is self.prev_scan:
                return
            dt, self.prev_scan, self.scan_ticks = self.scan_ticks * 0.05, self.last_scan, 0
            self.scans += 1
            along, across = project(direction)
            back, front, lo, hi = body(direction)
            zone = np.where((along > front - 0.05) & (along < front + ZONE) & (np.abs(across) < 1.6))[0]
            # Split the points, in scan order, wherever neighbours are far apart.
            # Skip pieces cut off by something nearer (e.g. a wall seen between
            # two obstacles): their ends are not real ends, so they look like
            # something moving as the obstacles move.
            a, d = across[zone], along[zone]
            splits = np.where(np.hypot(np.diff(a), np.diff(d)) > 0.08)[0] + 1
            groups = []
            r, n = scan, len(scan)
            for ca, cd, ci in zip(np.split(a, splits), np.split(d, splits), np.split(zone, splits)):
                if len(ci) and r[(ci[0] - 1) % n] > r[ci[0]] - 0.05 and r[(ci[-1] + 1) % n] > r[ci[-1]] - 0.05:
                    groups.append((ca, cd))
            travelled = progress(direction)
            for ca, cd in groups:
                if len(ca) < 3 or cd.max() - cd.min() > 0.1 or not MOVING_SIZE[0] < ca.max() - ca.min() < MOVING_SIZE[1]:
                    continue # Not (all of) a moving obstacle
                mid, where = (ca.min() + ca.max()) / 2, cd.min() + travelled
                path = next((p for p in self.paths if abs(p["where"] - where) < 0.1), None)
                if path is None:
                    self.paths.append({"where": where, "mid": mid, "half": (ca.max() - ca.min()) / 2, "speed": None, "speeds": [], "seen": 1, "last": self.scans,
                                       "heading": 0, "low": mid, "high": mid, "low_known": False, "high_known": False})
                    continue
                path["seen"] += 1
                path["last"] = self.scans
                v = (mid - path["mid"]) / dt
                path["half"] = min(max(path["half"], (ca.max() - ca.min()) / 2), 0.2)
                if abs(mid - path["mid"]) < 0.3 and abs(v) > 0.1:
                    path["speeds"] = path["speeds"][-9:] + [abs(v)]
                    path["speed"] = float(np.median(path["speeds"])) # Robust to the odd bad reading
                    if path["heading"] == 1 and v < 0:
                        path["high_known"] = True # Seen it turn round at the high end
                    if path["heading"] == -1 and v > 0:
                        path["low_known"] = True
                    path["heading"] = 1 if v > 0 else -1
                path["mid"], path["low"], path["high"] = mid, min(path["low"], mid), max(path["high"], mid)

        def safe_to_cross(direction):
            '''(along distance to the nearest moving obstacle's path, True if no moving
            obstacle will be in the robot's lane while the robot crosses its path).
            Where an obstacle has not yet been seen to turn round, it is assumed it
            could turn round at any moment. After waiting WAIT ticks, settle for
            the obstacle only just touching the lane.'''
            back, front, lo, hi = body(direction)
            travelled, speed = progress(direction), self.max_translate_velocity * self.cross_ratio
            nearest, safe, overlap = np.inf, True, 0.0
            for path in self.paths:
                where = path["where"] - travelled
                if path["speed"] is None and path["seen"] >= 10:
                    continue # Seen for a second without moving: not a moving obstacle
                if self.scans - path["last"] > 10:
                    continue # Not seen for a second: gone
                if where < front - 0.05 or where > front + WAIT_LINE + 0.1:
                    continue # Not the path about to be crossed
                nearest = min(nearest, where)
                if path["speed"] is None:
                    safe = False # Not measured yet
                    continue
                low = path["low"] if path["low_known"] else (path["mid"] if path["heading"] < 0 else -np.inf)
                high = path["high"] if path["high_known"] else (path["mid"] if path["heading"] > 0 else np.inf)
                x, h = path["mid"], path["heading"] or 1
                enter, leave = (where - front - MARGIN) / speed, (where - back + MARGIN + 0.05) / speed
                for t in np.arange(0.0, leave, 0.05):
                    x += h * path["speed"] * 0.05
                    if x > high:
                        x, h = 2 * high - x, -1
                    elif x < low:
                        x, h = 2 * low - x, 1
                    if t >= enter and x - path["half"] - MARGIN < hi and x + path["half"] + MARGIN > lo:
                        safe = False
                        overlap += 0.05
            return nearest, safe or (self.wait >= WAIT and overlap <= 0.1)

        def next_leg(reason):
            self.leg += 1
            self.turning = False
            self.wait = 0
            self.get_logger().info(f"{reason}: starting leg {self.leg}")

        if self.leg in (0, 1) and wall_ahead(legs[self.leg], WALL_DEPTH):
            next_leg(f"Wall ahead travelling {legs[self.leg]}")

        elif self.leg == 2 and in_gate():
            next_leg("Through the gate")

        elif self.leg == 3 and lane_blocked("front", STOP) and wall_ahead("front", WALL_DEPTH):
            self.move_2D(0.0, 0.0, 0.0)
            self.get_logger().info("Reached the end wall: course finished")
            raise SystemExit

        self.prev_dir = legs[self.leg]

        # Sectors in anticlockwise order, so the directions relative to the
        # travel direction are found by stepping round the ring.
        ring = ["front", "front_left", "left", "rear_left",
                "rear", "rear_right", "right", "front_right"]
        i = ring.index(self.prev_dir)
        ahead, left_side, behind, right_side = (ring[(i + k) % 8] for k in (0, 2, 4, 6))

        def drive(forward, sideways):
            '''Send a command, sped up to at least MIN_SPEED unless it is a stop.'''
            command = np.array((forward, sideways), float)
            size = np.linalg.norm(command)
            if 0 < size < MIN_SPEED:
                command *= MIN_SPEED / size
            self.last_cmd = command
            self.move_2D(command[0], command[1], 0.0)

        def go(*moves):
            '''Move along the sum of (direction, scale) pairs.'''
            drive(*sum(velocity[direction] * scale for direction, scale in moves))

        self.get_logger().debug(str(distance))

        ahead_blocked = lane_blocked(ahead, STOP, 0.0, HOLD)
        push = push_away()

        # Past the gate only the moving obstacles are in the way: wait short
        # of each one's path until it is safe to cross, instead of dodging
        # (they move faster than the robot).
        path_ahead, safe = np.inf, True
        if self.leg == 3:
            if self.leg_start is None:
                self.leg_start = self.pose[:2].copy()
            if progress(ahead) > 0.15: # Clear of the gate's walls
                track_moving(ahead)
            path_ahead, safe = safe_to_cross(ahead)
        at_path = path_ahead < body(ahead)[1] + WAIT_LINE

        if self.leg == 3 and self.crossing is not None and progress(ahead) < self.crossing:
            # Committed: never stop on a moving obstacle's path. If one moves
            # into the lane in front anyway, reverse out of its way.
            if lane_blocked(ahead, 0.05, 0.0, HOLD):
                self.crossing = None
                go((behind, cross_speed))
            else:
                go((ahead, cross_speed))

        elif self.leg == 3 and (ahead_blocked or (at_path and not safe)):
            self.wait += 1
            drive(0.0, 0.0) # Wait short of the path for it to be safe

        elif self.leg == 3 and at_path:
            # Safe: cross until the robot's rear is past the path
            self.wait = 0
            self.crossing = progress(ahead) + path_ahead - body(ahead)[0] + MARGIN + 0.05
            go((ahead, cross_speed))

        elif ahead_blocked or self.turning:
            # Dodge: strafe to the nearest lane the robot fits through,
            # keeping to the side already chosen while it still has a lane.
            # With no lane, creep closer (whatever is beside the robot may
            # then be behind it), then back away (looking further ahead, so
            # the obstacle stays in view) until a lane opens up.
            lanes = find_lane(ahead) if self.turning else {}
            if not self.turning:
                self.turning = True
                self.backing = False
                self.depth = LOOK
                lanes = find_lane(ahead)
                if lanes:
                    self.dir = 0.2 if min(lanes, key=lambda s: abs(lanes[s])) > 0 else -0.2 # Nearest lane
            side = 1 if self.dir > 0 else -1
            if side not in lanes and lanes:
                side = -side

            if not lane_blocked(ahead, self.depth, 0.0, HOLD):
                # In a free lane: centre in it, then carry on
                room_left, room_right = slack(ahead, self.depth)
                if abs(room_left - room_right) > 0.04: # Coarser than a scan's worth of movement
                    go((left_side if room_left > room_right else right_side, 1.0))
                else:
                    self.turning = False
                    go((ahead, 1.0))
            elif lanes:
                self.dir = 0.2 * side
                go((left_side if side > 0 else right_side, 1.0))
            elif not self.backing and not lane_blocked(ahead, 0.03, 0.0, HOLD):
                go((ahead, 1.0))
            elif lane_blocked(behind, 0.03, 0.0, HOLD):
                drive(0.0, 0.0) # Boxed in on all sides
            else:
                self.backing = True
                self.depth += 0.01 # Distance backed in one tick
                go((behind, 1.0))

        elif push is not None:
            drive(*(push * MIN_SPEED)) # Too close to something: ease away from it

        elif self.leg == 3:
            self.wait = 0
            go((ahead, 1.0))

        else:
            # Clearance from the robot's edges (the LiDAR is not at its centre)
            clear_left = distance[left_side] - body(left_side)[1]
            clear_right = distance[right_side] - body(right_side)[1]
            if clear_left + clear_right < CORRIDOR and max(clear_left, clear_right) < 0.9:
                # In a corridor: drift towards its middle
                drift = np.clip(clear_left - clear_right, -0.5, 0.5)
                go((ahead, 1.0), (left_side, drift))
            else:
                go((ahead, 1.0))
        ###### INSERT CODE HERE ######
                

def main(args=None):
    arg_parser = argparse.ArgumentParser(description="RB2301 final project")
    arg_parser.add_argument('--calibrate', action='store_true', help="Real robot: only print the Optitrack pose and the maze-frame pose twice a second, never drive")
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    print("Starting obstacle course")
    rclpy.init(args=ros_args)

    # Use lidar scan length to determine if in simulation or not
    lidar = LidarNode()
    while lidar.is_simulation is None:
        rclpy.spin_once(lidar)
    is_simulation = lidar.is_simulation
    lidar.destroy_node()

    # Start spinning the obstacle node and only stop once SystemExit error is raised within the node callback
    obstacle = ObstacleCourseNode(is_simulation, cli_args.calibrate)
    try:
        rclpy.spin(obstacle)
    except SystemExit:
        print("Shutting down")
    obstacle.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()