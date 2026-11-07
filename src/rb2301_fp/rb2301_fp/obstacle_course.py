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
        self.sim_time = None # simulation time of the latest odometry [s]

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
        self.sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
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
        # Map-as-you-go navigation. The robot maps what its LiDAR sees and heads for the reachable frontier farthest (along
        # the path) from where it started, until it finds the gate (a gap in a long thin wall). It lines up with it, goes
        # straight through, watches the moving obstacles beyond, then goes along the closer side wall past them, waiting
        # short of each while it is in front or coming this way, and stops once past N_MOVING of them.
        import heapq # standard library: priority queue for the shortest-path search
        import time  # standard library: monotonic clock
        import types # standard library: SimpleNamespace for the navigation state

        RES = 0.04          # grid cell size [m]
        SIZE = 16.0         # grid side [m], centred on the start pose
        RANGE = 6.0         # LiDAR range used for mapping [m]
        PLAN_PERIOD = 0.5   # replanning period [s]
        LOOKAHEAD = 0.3     # pure pursuit lookahead [m]
        ZONE_MEMORY = 30.0  # how long a cell stays marked as "something moves here" [s]
        SWEEP_MEMORY = 10.0 # how long the rest of a moving obstacle's line stays marked after it was last seen moving [s]
        GATE_WATCH = 2.0    # at the gate, stop this long to watch the moving obstacles beyond it [s]
        N_MOVING = 2        # moving obstacles past the gate: the robot stops once it is past this many
        WALL_GAP = 0.12     # past the gate: clearance kept between the robot and the side wall it goes along [m]
        # The robot only. Sim values are from the URDF; check the real ones. lidar_yaw_deg is where the LiDAR's 0 rad
        # points, counter-clockwise from robot forward (the start-up log line shows the bearing of the nearest object).
        ROBOT = {True:  dict(half_width=0.11, half_length=0.14, body_offset=0.07, lidar_x=0.114, lidar_yaw_deg=180.0),
                 False: dict(half_width=0.12, half_length=0.15, body_offset=0.0,  lidar_x=0.0,   lidar_yaw_deg=180.0)}

        if not hasattr(self, "nav"): # first tick: navigation state, kept on self between ticks
            nav = types.SimpleNamespace()
            nav.robot = ROBOT[bool(self.is_simulation)]
            vmax = self.max_translate_velocity
            nav.cruise, nav.dash, nav.turn_max = min(0.5, 0.8 * vmax), min(1.0, vmax), min(1.5, 2 * vmax)
            nav.d_strict = int(math.ceil((nav.robot["half_width"] + 0.02) / RES + 0.5)) # wall clearance [cells]
            nav.d_soft = nav.d_strict + 5 # paths are pushed away from walls up to this distance [cells]
            nav.n = n = int(SIZE / RES)
            nav.logodds = np.zeros((n, n), np.float32) # occupancy, indexed [ix, iy]
            nav.seen = np.zeros((n, n), bool)
            nav.seen_time = np.full((n, n), np.inf) # first time each cell was seen
            nav.dyn_time = np.full((n, n), -1e9) # last time something moving was seen in each cell
            nav.sweep_time = np.full((n, n), -1e9) # last time a cell was on the line something moves along (see sweep)
            nav.zone = np.zeros((n, n), bool) # cells where something moved recently or may move, grown by 2 cells
            nav.seen_zone = np.zeros((n, n), bool) # cells where something was seen moving recently, grown by 2 cells
            nav.origin = nav.start = None
            nav.scan, nav.scan_seq, nav.used_seq = None, 0, 0
            nav.points = None # latest scan in the world frame: x, y, is_dynamic
            nav.tracks = [] # moving objects in the latest scan: dict(p, v, r, moving)
            nav.gate_pos, nav.zone_entry = None, None # where the gate is; where the robot last went into a zone
            nav.gate_watch = None # time the robot, crossed and facing front, started watching
            nav.gate_time = None # time the robot found the gate just ahead
            nav.cross_time = None # time the robot crossed through the gate into the final section: moving obstacles are
                                  # only looked for and avoided from then on
            nav.lines, nav.passed = [], [] # moving obstacles' lines of travel (how far past the gate), and those passed
            nav.wall_lat, nav.wall_phase = None, "to wall" # "wall": the lane by the wall (sideways from the gate's centre)
            nav.wall_going, nav.wall_why = None, None # the obstacle line being crossed (committed), and the last decision
            nav.wall_still = None # (time, place) the robot was last seen moving, going to the wall
            nav.trail = [] # the robot's last metre of positions
            nav.gate_lineup = None # where it lines up with the gate
            nav.gate_phase = None # "line up", then "through" (see the gate steps in drive)
            nav.path = nav.goal = None
            nav.goal_final = nav.narrow = False
            nav.last_plan = nav.last_log = 0.0
            nav.in_zone = nav.crossed = False # crossed: through the gate, into the final section
            nav.travel = np.zeros(2) # smoothed direction of travel
            nav.last_c = None
            nav.old_scans = [] # recent scans (t, lidar x, lidar y, first beam angle, ranges) for the slow-obstacle test
            nav.now = 0.0

            def scan_cb(msg): # own copy of the scan, with the beam angles
                step = max(1, len(msg.ranges) // 400)
                r = np.asarray(msg.ranges, float)[::step]
                a = (msg.angle_min + msg.angle_increment * np.arange(0, len(msg.ranges), step))[:len(r)]
                valid = np.isfinite(r) & (r > max(msg.range_min, 0.08)) & (r < msg.range_max)
                nav.scan = (np.where(valid, r, 0.0), a, valid)
                nav.scan_seq += 1
            self.create_subscription(LaserScan, "scan", scan_cb, 2)
            self.nav = nav
        nav = self.nav
        log = self.get_logger()

        def dilate(mask, steps=1):
            '''Grow a boolean grid by steps cells (8-connected).'''
            for _ in range(steps):
                m = mask.copy()
                m[1:] |= mask[:-1]
                m[:-1] |= mask[1:]
                mask = m.copy()
                mask[:, 1:] |= m[:, :-1]
                mask[:, :-1] |= m[:, 1:]
            return mask

        def dijkstra(trav, start, goal=-1, cost=None):
            '''Shortest paths (in cells) over the True cells of trav, whose border must be False, from flat index start.
            Returns (dist, parent) as flat lists. Stops early once goal is reached.'''
            W = trav.shape[1]
            ok = trav.ravel().tolist()
            extra = cost.ravel().tolist() if cost is not None else None
            dist, parent = [math.inf] * len(ok), [-1] * len(ok)
            dist[start] = 0.0
            heap = [(0.0, start)]
            steps = [(1, 1.0), (-1, 1.0), (W, 1.0), (-W, 1.0), (W + 1, 1.414), (W - 1, 1.414), (1 - W, 1.414), (-1 - W, 1.414)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue
                if u == goal:
                    break
                for off, w in steps:
                    v = u + off
                    if ok[v]:
                        nd = d + (w * (1.0 + extra[v]) if extra else w)
                        if nd < dist[v]:
                            dist[v], parent[v] = nd, u
                            heapq.heappush(heap, (nd, v))
            return dist, parent

        def cell(x, y):
            return (np.floor((np.asarray(x) - nav.origin[0]) / RES).astype(int),
                    np.floor((np.asarray(y) - nav.origin[1]) / RES).astype(int))

        def integrate(x, y, h, now):
            '''Ray-cast the latest scan into the grid. Hits on cells known to be free, away from walls, are moving things.'''
            r, a, valid = nav.scan
            n = nav.n
            lx, ly = x + nav.robot["lidar_x"] * math.cos(h), y + nav.robot["lidar_x"] * math.sin(h)
            ang = h + math.radians(nav.robot["lidar_yaw_deg"]) + a
            ca, sa = np.cos(ang), np.sin(ang)
            hx, hy = lx + r * ca, ly + r * sa
            hi, hj = cell(hx, hy)
            hit = valid & (r <= RANGE) & (hi >= 0) & (hi < n) & (hj >= 0) & (hj < n)
            t = np.arange(0.0, RANGE, RES / 2)
            along = t[None, :] < np.where(valid, np.minimum(r, RANGE) - RES, 0.0)[:, None]
            fi, fj = cell((lx + t[None, :] * ca[:, None])[along], (ly + t[None, :] * sa[:, None])[along])
            inb = (fi >= 0) & (fi < n) & (fj >= 0) & (fj < n)
            miss = np.zeros((n, n), bool)
            miss[fi[inb], fj[inb]] = True
            dyn = np.zeros(len(r), bool)
            idx = np.nonzero(hit)[0]
            not_free = dilate(nav.logodds > -0.4) # cells with any neighbour not clearly free (walls, unseen space behind them)
            dyn[idx] = (nav.logodds[hi[idx], hj[idx]] < -1.2) & ~not_free[hi[idx], hj[idx]]
            # Also moving: a hit in front of where the scan 0.3 s ago saw free space, on all three neighbouring beams. This
            # catches slow obstacles, which the test above misses (they are always next to cells they occupied a moment ago),
            # and never fires on a static wall (an old beam towards a point on a wall ends at that wall)
            old = [s for s in nav.old_scans if s[0] <= now - 0.3]
            if old and len(old[-1][4]) == len(r) and len(r) > 1:
                _, olx, oly, oang0, orr = old[-1]
                dist = np.hypot(hx - olx, hy - oly)
                k_old = np.round(np.mod(np.arctan2(hy - oly, hx - olx) - oang0, 2 * math.pi) / float(a[1] - a[0])).astype(int)
                moved = hit.copy()
                for kk in (-1, 0, 1):
                    moved &= orr[(k_old + kk) % len(r)] > dist + 0.1 + 0.02 * dist
                dyn |= moved
            nav.old_scans = [s for s in nav.old_scans if s[0] > now - 1.0] + [(now, lx, ly, ang[0], np.where(valid, r, np.inf))]
            stat = hit & ~dyn
            miss[hi[hit], hj[hit]] = False
            nav.logodds[miss] -= 0.4
            nav.logodds[hi[stat], hj[stat]] += 0.85
            np.clip(nav.logodds, -2.0, 3.5, out=nav.logodds)
            nav.seen |= miss
            nav.seen[hi[hit], hj[hit]] = True
            nav.seen_time[nav.seen & np.isinf(nav.seen_time)] = now
            nav.points = (hx[valid], hy[valid], dyn[valid])
            # Moving objects: runs of neighbouring hits that hold moving hits. Only the leading edge of an object moving
            # sideways tests as moving (the rest of it is where it already was a moment ago), so an object is its whole
            # run, kept to the cells where something moved recently. Without this, a 0.3 m wide panel looks 5 cm wide
            # and the robot sets off as soon as its leading edge has passed, into the rest of it
            idx = np.nonzero(hit)[0]
            objects = []
            if len(idx) >= 2:
                px, py, pd, pi, pj = hx[idx], hy[idx], dyn[idx], hi[idx], hj[idx]
                pz = pd | nav.seen_zone[pi, pj] # (not where things only may move: that reaches into walls)
                run = np.concatenate([[0], np.cumsum(np.hypot(np.diff(px), np.diff(py)) > 0.15)])
                for s in np.unique(run[pd]):
                    md = (run == s) & pd
                    if md.sum() < 2:
                        continue
                    cx, cy = px[md].mean(), py[md].mean()
                    body = md | ((run == s) & pz & (np.hypot(px - cx, py - cy) < 0.5))
                    objects.append(make_object(np.stack([px[body], py[body]], 1), (pi[md], pj[md]), now))
            track(objects, now)

        def make_object(pts, cells, now):
            '''Object from its LiDAR points: centre, long axis e, half length a along e and half width b across it.'''
            p = pts.mean(0)
            d = pts - p
            weak = len(pts) < 3 or np.ptp(d, 0).max() < 0.04 # too few points to tell which way it is long
            e = np.array([1.0, 0.0]) if weak else np.linalg.eigh(d.T @ d)[1][:, 1]
            a = float(np.clip(np.abs(d @ e).max(), 0.05, 0.5)) # (never more than 1 m long: then it has taken in some wall)
            b = float(np.clip(np.abs(d @ np.array([-e[1], e[0]])).max(), 0.04, 0.3))
            return dict(p=p, v=np.zeros(2), vdir=None, still=None, origin=p.copy(), e=e, a=a, b=b, weak=weak, moving=False, n=0, since=None, t=now, seen=now, cells=cells)

        def sweep(o, now):
            '''Mark the line a moving object moves along, both ways up to the walls, as somewhere it moves. Until it has been
            seen to turn around, it may go all the way: the robot must not cross the part of its track it has not been seen in
            yet (e.g. hidden behind something) as if nothing moved there.'''
            w, n = o["vdir"], perp(o["vdir"])
            across = o["a"] * abs(float(o["e"] @ n)) + o["b"] * abs(float(perp(o["e"]) @ n)) # its half width across the line
            along = o["a"] * abs(float(o["e"] @ w)) + o["b"] * abs(float(perp(o["e"]) @ w))
            offs = np.arange(-across, across + RES / 2, RES)
            ks = np.arange(0.0, 1.5, RES)
            for sgn in (1.0, -1.0):
                line = o["p"] + sgn * (along + ks)[:, None] * w # centre line, from the end of the object outwards
                li, lj = cell(line[:, 0], line[:, 1])
                if not ((li >= 0) & (li < nav.n) & (lj >= 0) & (lj < nav.n)).all():
                    continue
                wall = nav.logodds[li, lj] > 0.6 # first wall on the line (its own cells marked as moving do not count)
                wall &= (now - nav.dyn_time[li, lj]) >= ZONE_MEMORY
                end = int(np.argmax(wall)) - 2 if wall.any() else len(ks)
                if end <= 0:
                    continue
                pts = line[:end, None, :] + offs[None, :, None] * n # band as wide as the object
                pi, pj = cell(pts[..., 0].ravel(), pts[..., 1].ravel())
                inb = (pi >= 0) & (pi < nav.n) & (pj >= 0) & (pj < nav.n)
                pi, pj = pi[inb], pj[inb]
                free = nav.logodds[pi, pj] < 0.6 # never over a wall
                nav.sweep_time[pi[free], pj[free]] = now

        def track(objects, now):
            '''Match objects to the previous scan's, estimate their velocity, and mark the cells moving ones are in.'''
            unmatched = list(nav.tracks)
            for o in objects:
                prev = [q for q in unmatched if np.linalg.norm(q["p"] - o["p"]) < 0.4]
                if prev:
                    q = min(prev, key=lambda q: np.linalg.norm(q["p"] - o["p"]))
                    unmatched = [u for u in unmatched if u is not q]
                    step = (o["p"] - q["p"]) / max(now - q["t"], 0.02)
                    if np.linalg.norm(step) > 1.0: # faster than anything on the course: matched the wrong thing, keep the old one
                        step = q["v"]
                    o["v"] = 0.5 * q["v"] + 0.5 * step
                    o["n"] = q["n"] + (np.linalg.norm(o["v"]) > 0.1) # scans in a row seen moving
                    o["origin"] = q["origin"] # where it was first seen: something really moving gets away from there, unlike
                    # a wall flickering at close range
                    o["moving"] = q["moving"] or (o["n"] >= 3 and np.linalg.norm(o["p"] - o["origin"]) > 0.3)
                    if nav.crossed and float((o["p"] - nav.gate_pos) @ nav.gate_dir) < 0.15: # not past the gate: not one
                        o["moving"] = False # of the moving obstacles (e.g. the gate's walls flickering close up)
                    o["since"] = q["since"]
                    sp = float(np.linalg.norm(o["v"]))
                    o["vdir"] = o["v"] / sp if sp > 0.15 else q["vdir"] # which way along its track it moves (last seen)
                    # Still since when: moving obstacles only stop for a moment, to turn around. Still for long, it is not one
                    o["still"] = (q["still"] or now) if sp < 0.06 else None
                    if o["still"] is not None and now - o["still"] > 2.0:
                        o["moving"], o["n"], o["since"], o["vdir"] = False, 0, None, None
                    if o["weak"]:
                        o["e"] = q["e"]
                    if abs(float(o["e"] @ q["e"])) > 0.9: # same orientation: it is at least as big as it looked before
                        o["a"], o["b"] = max(o["a"], 0.97 * q["a"]), max(o["b"], 0.97 * q["b"])
                if o["moving"]:
                    o["since"] = o["since"] or now
                    nav.dyn_time[o["cells"]] = now
                    if nav.crossed and o["vdir"] is not None and now - o["since"] >= 1.0 and np.linalg.norm(o["v"]) > 0.15: # surely moving
                        sweep(o, now)
            # One object seen in pieces counts once: of moving ones closer than 0.35 m, only the biggest
            seen_movers = sorted([o for o in objects if o["moving"]], key=lambda o: -o["a"])
            kept = []
            for o in seen_movers:
                if any(np.linalg.norm(o["p"] - k["p"]) < 0.35 for k in kept):
                    objects = [x for x in objects if x is not o] # (by identity: == on these dicts compares arrays)
                else:
                    kept.append(o)
            for q in unmatched: # keep moving objects that dropped out of view (e.g. behind another), where they would be now
                if q["moving"] and now - q["seen"] < 3.0:
                    sp, dt = float(np.linalg.norm(q["v"])), now - q["t"]
                    if sp > 0.1:
                        w = q["v"] / sp
                        lo, hi = reach(q, w, now)
                        x = sp * dt
                        if x > hi and hi - lo > 1e-3: # it has turned around at the end of its track
                            x, q["v"] = hi - min(x - hi, hi - lo), -q["v"]
                            q["vdir"] = -w
                        q["p"] = q["p"] + x * w
                    q["t"] = now
                    same = lambda k: np.linalg.norm(q["p"] - k["p"]) < 0.5 or (k["vdir"] is not None and abs(float((q["p"] - k["p"]) @ perp(k["vdir"]))) < 0.2)
                    if not any(same(k) for k in kept): # (not if it is back in view already: near it, or on its line of travel)
                        objects.append(q)
            nav.tracks = objects
            nav.seen_zone = dilate((now - nav.dyn_time) < ZONE_MEMORY, 2) # where things have been seen moving
            nav.zone = nav.seen_zone | dilate((now - nav.sweep_time) < SWEEP_MEMORY, 2) # and where they may move

        def nearest(trav, i, j):
            '''Nearest traversable cell to local (i, j), within 6 cells.'''
            H, W = trav.shape
            best = None
            for di in range(-6, 7):
                for dj in range(-6, 7):
                    a, b = i + di, j + dj
                    if 0 <= a < H and 0 <= b < W and trav[a, b] and (best is None or di * di + dj * dj < best[0]):
                        best = (di * di + dj * dj, a, b)
            return None if best is None else (best[1], best[2])

        def plan_path(c):
            '''Pick the goal (farthest reachable frontier from the start) and plan a path to it.'''
            ii, jj = np.nonzero(nav.seen)
            if len(ii) == 0:
                return
            i0, i1 = max(ii.min() - 12, 0), min(ii.max() + 13, nav.n)
            j0, j1 = max(jj.min() - 12, 0), min(jj.max() + 13, nav.n)
            lo, seen = nav.logodds[i0:i1, j0:j1], nav.seen[i0:i1, j0:j1]
            moving = nav.zone[i0:i1, j0:j1]
            occ = (lo > 0.6) & ~((now - nav.dyn_time[i0:i1, j0:j1]) < ZONE_MEMORY) & ~((now - nav.sweep_time[i0:i1, j0:j1]) < SWEEP_MEMORY)
            dist, grow = np.full(occ.shape, nav.d_soft, np.int16), occ
            for k in range(nav.d_soft): # distance to the nearest wall, in cells
                dist[grow & (dist > k)] = k
                grow = dilate(grow)
            free = seen & ((lo < 0.0) | moving)
            frontier = free & dilate(~seen)
            f = frontier.astype(np.int8)
            nb = f.copy()
            nb[1:] += f[:-1]; nb[:-1] += f[1:]
            nb2 = nb.copy()
            nb2[:, 1:] += nb[:, :-1]; nb2[:, :-1] += nb[:, 1:]
            frontier &= nb2 >= 3 # drop specks
            cost = 3.0 * np.clip((nav.d_soft - dist) / (nav.d_soft - nav.d_strict + 1.0), 0, 1) ** 2
            cost = cost + 2.0 * moving # cross where things move the shortest way (straight across), not along them
            H, W = occ.shape
            local = lambda p: tuple(np.floor((p - nav.origin) / RES).astype(int) - (i0, j0))
            # Once at the gate, the goal is always past it: never back out of it (when part of the room beyond looks blocked
            # for a moment, e.g. by things moving, the robot holds where it is instead)
            past_gate = np.ones((H, W), bool)
            if nav.gate_time is not None:
                gi, gj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
                past_gate = ((nav.origin[0] + (gi + 0.5) * RES - nav.gate_pos[0]) * nav.gate_dir[0]
                             + (nav.origin[1] + (gj + 0.5) * RES - nav.gate_pos[1]) * nav.gate_dir[1]) > 0.3

            def choose(d_min):
                trav = free & (dist >= d_min)
                trav[0], trav[-1], trav[:, 0], trav[:, -1] = False, False, False, False
                rc = nearest(trav, *local(c))
                if rc is None:
                    return None
                sc = nearest(trav, *local(nav.start)) or rc
                ds = np.array(dijkstra(trav, sc[0] * W + sc[1])[0]).reshape(H, W)
                if not np.isfinite(ds[rc]): # start cut off at this clearance: measure from the robot instead
                    ds = np.array(dijkstra(trav, rc[0] * W + rc[1])[0]).reshape(H, W)
                cand = frontier & trav & np.isfinite(ds) & (ds >= ds[rc] - 0.3 / RES) & past_gate # never go back
                final = not cand.any()
                if final:
                    cand = trav & np.isfinite(ds) & past_gate
                    if not cand.any():
                        return None # nothing reachable past the gate right now: hold
                score = np.where(cand, ds, -1.0)
                g = np.unravel_index(int(np.argmax(score)), score.shape)
                if nav.goal is not None and final == nav.goal_final: # keep the goal unless clearly worse (final ones too: two far corners must not take turns)
                    og = local(nav.goal)
                    if 0 <= og[0] < H and 0 <= og[1] < W and cand[og] and ds[og] + 0.5 / RES > ds[g]:
                        g = og
                return dict(trav=trav, rc=rc, g=g, ds=ds[g], final=final, narrow=d_min < nav.d_strict)

            plan = choose(nav.d_strict)
            if plan is None or plan["final"] or math.dist(plan["g"], plan["rc"]) < 1.0 / RES:
                relaxed = choose(nav.d_strict - 1) # a narrow door may only open with less clearance
                if relaxed is not None and (plan is None or relaxed["ds"] > plan["ds"] + 0.3 / RES):
                    plan = relaxed
            if plan is None:
                nav.path = None
                return
            rc, g = plan["rc"], plan["g"]
            _, parent = dijkstra(plan["trav"], rc[0] * W + rc[1], g[0] * W + g[1], cost)
            cells, u = [], g[0] * W + g[1]
            while u != -1 and len(cells) < 5000:
                cells.append(u)
                u = parent[u] if u != rc[0] * W + rc[1] else -1
            cells = np.array(cells[::-1])
            pts = nav.origin + (np.stack([cells // W + i0, cells % W + j0], 1) + 0.5) * RES
            nav.path = np.vstack([c, pts])
            nav.goal, nav.goal_final, nav.narrow = pts[-1], plan["final"], plan["narrow"]

        def perp(u):
            return np.array([-u[1], u[0]])

        def doorway(p, u):
            '''Is p (going along unit u) in a gate: a gap narrower than 0.9 m in a long thin wall (running sideways from it
            0.4 m or more on each side, 1 m or more in all), with open space just before and after it (not a corridor)?
            Returns (the direction straight through it, square to the wall the way of u; the middle of the gap), or None.'''
            n = perp(u)
            ks = np.arange(0.0, 0.8, RES)
            def occ(q):
                qi, qj = cell(q[:, 0], q[:, 1])
                ok = (qi >= 0) & (qi < nav.n) & (qj >= 0) & (qj < nav.n)
                return ok & (nav.logodds[np.clip(qi, 0, nav.n - 1), np.clip(qj, 0, nav.n - 1)] > 0.6)
            def clear(q0, d): # free distance from q0 along d (up to 0.8 m)
                hit = occ(q0 + ks[:, None] * d)
                return float(ks[np.argmax(hit)]) if hit.any() else 0.8
            dl, dr = clear(p, n), clear(p, -n)
            if dl >= 0.8 or dr >= 0.8 or dl + dr > 0.9:
                return None
            runs = []
            for d, edge in ((n, dl), (-n, dr)): # the walls on both sides run sideways, away from the gap, straight and thin
                run = np.arange(edge, edge + 1.0, RES)
                wall = np.zeros(len(run), bool)
                for du in (-2 * RES, -RES, 0.0, RES, 2 * RES): # walls are thin: allow for where exactly they are
                    wall |= occ(p + du * u + run[:, None] * d)
                gaps = np.nonzero(~wall[:-1] & ~wall[1:])[0] # how far it goes on (single missing cells allowed)
                length = (gaps[0] if len(gaps) else len(run)) * RES
                if length < 0.4:
                    return None
                seg = run[:max(int(length / RES), 1)]
                beside = occ(p + 0.15 * u + seg[:, None] * d) | occ(p - 0.15 * u + seg[:, None] * d)
                if beside.mean() > 0.3: # not thin: a bin, drawer or box, not a wall
                    return None
                runs.append(length)
            if sum(runs) < 1.0: # a gate is a gap in a long wall
                return None
            if not all(clear(p + su * u, n) + clear(p + su * u, -n) > 1.0 for su in (-0.35, 0.35)): # open before and after
                return None
            # Which way the wall runs: the line through the gap that lies on the most wall (both its sides are on it).
            # Tried within 45 deg of square to the way the path goes through
            base = math.atan2(n[1], n[0])
            best, best_count = n, -1
            for da in np.radians(np.arange(-45, 46, 1)):
                w = np.array([math.cos(base + da), math.sin(base + da)])
                q = p + np.arange(-1.0, 1.0, RES / 2)[:, None] * w
                count = int(occ(q).sum())
                if count > best_count:
                    best, best_count = w, count
            ts = np.arange(-1.0, 1.0, RES / 2)
            on = occ(p + ts[:, None] * best)
            if min(on[ts > 0].sum(), on[ts < 0].sum()) * RES / 2 < 0.25: # both sides of the gap on that one straight wall
                return None
            through = perp(best)
            through = through if through @ u > 0 else -through
            ns = perp(through) # the middle of the gap, square to the way through it
            return through, p + ns * (clear(p, ns) - clear(p, -ns)) / 2

        def wall_follow(c, h, now):
            '''Past the gate: past the moving obstacles along the closer side wall, facing front. Their tracks end at the
            walls, so there each one only has to be let by: wait short of its path while it is in front or coming this way,
            else go across. Nearest first. Returns (direction, speed, heading error to face front [deg], what it is doing).'''
            hl, hw = nav.robot["half_length"], nav.robot["half_width"]
            u, n = nav.gate_dir, perp(nav.gate_dir) # straight on through the gate, and to its left
            along, lat = float((c - nav.gate_pos) @ u), float((c - nav.gate_pos) @ n)
            face = wrap_deg(math.degrees(math.atan2(u[1], u[0]) - h))
            if nav.wall_lat is None: # which wall is closer, once: the free distance each way, square to the way through
                ks = np.arange(1, int(2.0 / RES)) * RES
                room = {}
                for sgn in (1.0, -1.0):
                    q = c + ks[:, None] * (sgn * n)
                    pi, pj = cell(q[:, 0], q[:, 1])
                    pi, pj = np.clip(pi, 0, nav.n - 1), np.clip(pj, 0, nav.n - 1)
                    wall = (nav.logodds[pi, pj] > 0.6) & ((now - nav.dyn_time[pi, pj]) >= ZONE_MEMORY) & ((now - nav.sweep_time[pi, pj]) >= SWEEP_MEMORY)
                    room[sgn] = float(ks[np.argmax(wall)]) if wall.any() else 2.0
                sgn = 1.0 if room[1.0] <= room[-1.0] else -1.0
                nav.wall_lat = lat + sgn * max(0.0, room[sgn] - hw - WALL_GAP)
                log.info(f"Final zone: to the {'left' if sgn > 0 else 'right'} wall, {room[sgn]:.2f} m away (the other is {room[-sgn]:.2f} m)")
            off = nav.wall_lat - lat # how far the robot is from its lane by the wall
            if nav.wall_phase == "to wall":
                if nav.wall_still is None or math.dist(c, nav.wall_still[1]) > 0.02:
                    nav.wall_still = (now, c.copy()) # (when it last moved: slowing by the wall, it may stop a little short)
                if abs(off) > 0.06 and now - nav.wall_still[0] < 1.5:
                    return n * np.sign(off), min(0.25, max(0.15, 2.0 * abs(off))), face, "to wall"
                nav.wall_phase, nav.wall_lat = "along", lat # the lane is where it got to
                log.info(f"At the wall ({c[0]:.2f}, {c[1]:.2f}): going along it")
            fwd = u + n * float(np.clip(2.0 * off, -0.5, 0.5)) # forward, keeping to the lane
            fwd = fwd / np.linalg.norm(fwd)
            ahead = [d for d in sorted(nav.lines) if d not in nav.passed] # obstacles' lines still to cross, nearest first
            if nav.wall_going is not None: # crossing one: never stop in its path
                if nav.wall_going in ahead:
                    return fwd, nav.dash, face, "going"
                nav.wall_going = None
            # Where to stop short of the next one: its line, or (not known yet) where things were seen moving on the lane
            stop, line = None, None
            if ahead:
                line = ahead[0]
                stop = line - hl - WALL_GAP - 0.03
            ts = np.arange(RES, 1.5, RES)
            q = c + ts[:, None] * u
            pi, pj = cell(q[:, 0], q[:, 1])
            moved = nav.seen_zone[np.clip(pi, 0, nav.n - 1), np.clip(pj, 0, nav.n - 1)]
            if moved.any() and (stop is None or along + ts[np.argmax(moved)] - hl - 0.05 < stop - 0.1):
                stop, line = along + ts[np.argmax(moved)] - hl - 0.05, None # something moving there not known yet: wait
            if stop is None: # nothing (known to be) moving ahead: on, gently
                return fwd, 0.5 * nav.cruise, face, "along"
            go, why = False, "not seen yet"
            if line is not None:
                movers = [o for o in nav.tracks if o["moving"] and abs(float((o["p"] - nav.gate_pos) @ u) - line) < 0.3]
                if movers:
                    go, why = True, "moving away"
                    for o in movers:
                        plat, vlat = float((o["p"] - nav.gate_pos) @ n), float(o["v"] @ n)
                        half = o["a"] * abs(float(o["e"] @ n)) + o["b"] * abs(float(perp(o["e"]) @ n)) # its half length across
                        if abs(plat - nav.wall_lat) < half + hw + 0.15:
                            go, why = False, "in front"
                        elif (nav.wall_lat - plat) * vlat > 0 and abs(vlat) > 0.05:
                            go, why = False, f"coming this way at {abs(vlat):.2f} m/s"
            k = len(nav.passed) + 1
            if (k, why.split(" at ")[0]) != nav.wall_why: # (once each time the decision changes, not with every speed)
                nav.wall_why = (k, why.split(" at ")[0])
                log.info(f"Obstacle {k} {why}: " + ("going across" if go else f"waiting short of it"))
            if go:
                if along >= stop - 0.03: # at the stopping point (or past it) and clear: committed
                    nav.wall_going = line
                return fwd, nav.dash, face, "go"
            gap = stop - along
            if gap > 0.02:
                return fwd, float(np.clip(1.5 * gap, 0.12, nav.cruise)), face, "to stop"
            return fwd, 0.0, face, "waiting"

        def reach(o, w, now):
            '''Range (lo <= 0 <= hi) along unit w that the object's centre moves in, relative to where it is now: it turns
            around at the far ends of the cells it has been seen moving through.'''
            ci, cj = cell(o["p"][0], o["p"][1])
            R = int(1.5 / RES)
            i0, j0 = max(int(ci) - R, 0), max(int(cj) - R, 0)
            ii, jj = np.nonzero((now - nav.dyn_time[i0:int(ci) + R + 1, j0:int(cj) + R + 1]) < ZONE_MEMORY)
            rel = nav.origin + (np.stack([ii + i0, jj + j0], 1) + 0.5) * RES - o["p"]
            e, n = o["e"], perp(o["e"])
            half = lambda u: o["a"] * abs(float(e @ u)) + o["b"] * abs(float(n @ u)) # object's half size along u
            s_ = np.sort(rel[np.abs(rel @ perp(w)) < half(perp(w)) + 0.08] @ w)
            if len(s_) == 0:
                return 0.0, 0.0
            k = int(np.argmin(np.abs(s_)))
            if abs(s_[k]) > 0.2:
                return 0.0, 0.0
            breaks = np.nonzero(np.diff(s_) > 0.12)[0] # the stretch it moves along: no gaps, starting from where it is
            lo = s_[breaks[breaks < k].max() + 1] if (breaks < k).any() else s_[0]
            hi = s_[breaks[breaks >= k].min()] if (breaks >= k).any() else s_[-1]
            return min(0.0, lo - RES / 2 + half(w)), max(0.0, hi + RES / 2 - half(w))

        def brake_factor(c, h, direction, dashing):
            '''Speed factor in [0, 1] from the raw scan in the direction of motion (moving things ignored while dashing),
            and the offset of the closest point in the way.'''
            px, py, dyn = nav.points
            keep = np.ones(len(px), bool)
            if dashing: # never stop inside a crossing because of the moving things themselves (but do for walls)
                pi, pj = cell(px, py)
                near = np.zeros(len(px), bool)
                for o in nav.tracks:
                    if o["moving"]:
                        near |= np.hypot(px - o["p"][0], py - o["p"][1]) < o["a"] + 0.3
                keep = ~dyn & ~(near & nav.zone[np.clip(pi, 0, nav.n - 1), np.clip(pj, 0, nav.n - 1)])
            rx, ry = px[keep] - c[0], py[keep] - c[1]
            par = direction[0] * rx + direction[1] * ry
            perp = np.abs(direction[0] * ry - direction[1] * rx)
            fwd = abs(direction[0] * math.cos(h) + direction[1] * math.sin(h))
            hl, hw = nav.robot["half_length"], nav.robot["half_width"]
            along, across = hl * fwd + hw * (1 - fwd), hw * fwd + hl * (1 - fwd)
            # Only what is ahead of the front: a corner right beside the robot would otherwise stop it for good
            sel = np.nonzero((par > along - 0.03) & (perp < across + 0.01))[0]
            if len(sel) == 0:
                return 1.0, None
            k = sel[np.argmin(par[sel])]
            return float(np.clip((par[k] - along - 0.01) / 0.2, 0.0, 1.0)), np.array([rx[k], ry[k]])

        def drive(c, h, now):
            if nav.path is None or len(nav.path) < 2:
                self.move_2D(0.0, 0.0, 0.0)
                return
            P = nav.path
            d = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
            i = int(np.argmin(d))
            far = np.nonzero(d[i:] >= LOOKAHEAD)[0]
            vec = (P[i + far[0]] if len(far) else P[-1]) - c
            norm = math.hypot(*vec)
            direction = vec / norm if norm > 1e-3 else np.zeros(2)
            to_goal = math.hypot(*(P[-1] - c))
            speed = nav.cruise * (0.6 if nav.narrow else 1.0) * float(np.clip(to_goal / 0.4, 0.2, 1.0))
            if nav.goal_final and to_goal < 0.05:
                speed = 0.0
            # Look for the gate on the path just ahead, and on the stretch just driven (in case it was missed going through)
            if nav.trail == [] or math.dist(c, nav.trail[-1]) > 0.04:
                nav.trail = (nav.trail + [c.copy()])[-25:] # the last metre driven
            if nav.gate_time is None:
                hl = nav.robot["half_length"]
                ahead_p = P[i:]
                arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(ahead_p, axis=0).T))])
                ahead_p = ahead_p[arc <= hl + 0.4] # (further out, corners with a bin or box beside them can look like one)
                for line in (ahead_p, np.array(nav.trail[::-1])):
                    for k in range(len(line)):
                        tang = line[min(k + 3, len(line) - 1)] - line[max(k - 3, 0)]
                        if line is not ahead_p:
                            tang = -tang # the trail runs backwards from the robot
                        found = doorway(line[k], tang / np.linalg.norm(tang)) if np.linalg.norm(tang) > 1e-6 else None
                        if found is not None:
                            through, middle = found
                            nav.gate_time, nav.gate_pos, nav.gate_dir, nav.gate_phase = now, middle, through, "line up"
                            # Line up on the centre line where the robot is now if it is already closer than the usual
                            # spot (never back up to it); with its front already in the gap, just go on through
                            along = float((middle - c) @ through) # how far the robot is before the middle of the gap
                            nav.gate_lineup = middle - through * min(along, hl + 0.2)
                            if along < hl + 0.02:
                                nav.gate_phase = "through"
                            log.info(f"At the gate ({middle[0]:.2f}, {middle[1]:.2f}): " + ("lining up with it, facing front" if nav.gate_phase == "line up" else "already in it, going on through"))
                            break
                    if nav.gate_time is not None:
                        break
            # The gate, in three steps: line up (a robot length before it, on its centre line, turned to face front: square to
            # its wall), go straight through along the centre line still facing front, then once all of the robot is through
            # (crossed into the final section) stop and watch the moving obstacles for GATE_WATCH. Moving obstacles are only
            # looked for and avoided from then on. (Turning in or just past the gate, a corner catches on its walls)
            gate_ctl = nav.gate_time is not None and nav.gate_watch is None # the gate steps are in charge
            if gate_ctl:
                hl = nav.robot["half_length"]
                line_up = nav.gate_lineup
                past = nav.gate_pos + nav.gate_dir * (hl + 0.12) # its back just clear of the gate
                face_err = wrap_deg(math.degrees(math.atan2(nav.gate_dir[1], nav.gate_dir[0])) - math.degrees(h))
                if nav.cross_time is None and float((c - past) @ nav.gate_dir) > -0.02:
                    nav.cross_time, nav.crossed = now, True
                    for o in nav.tracks: # start afresh: only what is seen moving from now on counts
                        o.update(moving=False, n=0, since=None, vdir=None, origin=o["p"].copy())
                    log.info(f"Crossed into the final section at ({c[0]:.2f}, {c[1]:.2f}), heading {math.degrees(h):.0f} deg")
                if nav.cross_time is None and nav.gate_phase == "line up":
                    if (math.dist(c, line_up) < 0.05 and abs(face_err) < 4.0) or now - nav.gate_time > 8.0:
                        nav.gate_phase = "through"
                        log.info(f"Lined up with the gate (heading {math.degrees(h):.0f} deg): going straight through")
                if nav.cross_time is not None and (abs(face_err) < 4.0 or now - nav.cross_time > 4.0):
                    nav.gate_watch, gate_ctl = now, False
                    log.info(f"Facing front (heading {math.degrees(h):.0f} deg): watching for moving obstacles for {GATE_WATCH:.0f} s")
            mode = ("clear",)
            wall_ctl = nav.gate_watch is not None and now - nav.gate_watch >= GATE_WATCH # along the wall, from the watch on
            if gate_ctl or (nav.gate_watch is not None and now - nav.gate_watch < GATE_WATCH):
                mode = ("wait", 0.0) # (the gate steps set the motion; watching: stay put)
            if mode[0] == "clear": # look before entering space just seen, so anything moving there shows itself
                near = P[i:][d[i:] < 0.6 + nav.robot["half_length"]]
                pi, pj = cell(near[:, 0], near[:, 1])
                fresh = now - nav.seen_time[np.clip(pi, 0, nav.n - 1), np.clip(pj, 0, nav.n - 1)] < 1.5
                if fresh.any():
                    mode = ("wait", d[i:][d[i:] < 0.6 + nav.robot["half_length"]][np.argmax(fresh)] - nav.robot["half_length"] - 0.05)
            if mode[0] == "wait":
                speed = min(speed, max(0.0, 1.5 * mode[1]))
            if norm > 1e-3:
                brake, block = brake_factor(c, h, direction, False)
                if brake < 0.2: # blocked: slide along the obstacle instead of pushing into it
                    away = block / np.linalg.norm(block)
                    slide = direction - (direction @ away) * away
                    if np.linalg.norm(slide) > 0.3:
                        direction = slide / np.linalg.norm(slide)
                        brake = 0.5 * brake_factor(c, h, direction, False)[0]
                speed *= brake
            err = wrap_deg(math.degrees(math.atan2(direction[1], direction[0]) - h)) if norm > 1e-3 else 0.0
            if gate_ctl: # the gate steps: head for the step's spot, facing front (the robot can drive sideways)
                err = face_err
                spot = past if nav.cross_time is not None or nav.gate_phase == "through" else line_up
                gap = float(np.linalg.norm(spot - c))
                if nav.gate_phase == "through" and nav.cross_time is None: # steady all the way through (slowing down near
                    direction, speed = (spot - c) / max(gap, 1e-6), 0.5 * nav.cruise # the end, it may stop short of it)
                elif gap > (0.02 if nav.cross_time is None else 0.06): # lining up; once crossed, only if it has drifted
                    direction = (spot - c) / gap
                    speed = min(0.25, max(0.12, 2.0 * gap)) # (slower may not move it at all)
                else:
                    speed = 0.0
            if wall_ctl:
                direction, speed, err, doing = wall_follow(c, h, now)
                mode = (doing,)
                if speed > 0: # (stop for walls, but not for the moving obstacles)
                    speed *= brake_factor(c, h, direction, True)[0]
            turn = float(np.clip(2.0 * math.radians(err), -nav.turn_max, nav.turn_max)) # face the direction of travel
            if (gate_ctl or wall_ctl) and abs(err) > 2.0: # (slower than this, the robot may not turn at all)
                turn = math.copysign(max(abs(turn), 0.5), err)
            if abs(err) > 45 and not wall_ctl:
                speed *= 0.5
            vx, vy = direction * speed
            bo = nav.robot["body_offset"] # command base_footprint so the body centre moves as asked
            vx, vy = vx + turn * bo * math.sin(h), vy - turn * bo * math.cos(h)
            self.move_2D(math.cos(h) * vx + math.sin(h) * vy, -math.sin(h) * vx + math.cos(h) * vy, turn)
            check_finish(c, now, mode[0])

        def check_finish(c, now, mode):
            '''Done once the robot is past the moving obstacles, nothing moves ahead and nowhere is left to go.'''
            if nav.last_c is not None and np.linalg.norm(c - nav.last_c) > 0.02:
                nav.travel = 0.8 * nav.travel + 0.2 * (c - nav.last_c) / np.linalg.norm(c - nav.last_c)
                nav.last_c = c
            elif nav.last_c is None:
                nav.last_c = c
            ci, cj = cell(c[0], c[1])
            k = int(nav.robot["half_length"] / RES)
            # Only where things were seen moving: where they only may move (see sweep) can reach into the end of the course
            inside = bool(nav.seen_zone[max(ci - k, 0):ci + k + 1, max(cj - k, 0):cj + k + 1].any())
            # Moving obstacles' lines of travel (across the way), as how far past the gate they are; passed once all of the
            # robot is 15 cm past one. Past N_MOVING of them: stop, done
            if nav.crossed:
                for o in nav.tracks:
                    if o["moving"] and o["vdir"] is not None and abs(float(o["vdir"] @ nav.gate_dir)) < 0.5: # moving across the way
                        d = float((o["p"] - nav.gate_pos) @ nav.gate_dir)
                        if d > 0.15 and now - o["since"] > 1.0 and not any(abs(d - e) < 0.35 for e in nav.lines): # (seen moving a while)
                            nav.lines.append(d)
                            log.info(f"Moving obstacle {len(nav.lines)}: moves across {d:.2f} m past the gate")
                through = float((c - nav.gate_pos) @ nav.gate_dir)
                for d in sorted(nav.lines):
                    if d not in nav.passed and through > d + nav.robot["half_length"] + 0.15:
                        nav.passed.append(d)
                        log.info(f"Past moving obstacle {len(nav.passed)} of {N_MOVING} (its line {d:.2f} m past the gate)")
                if len(nav.passed) >= N_MOVING:
                    self.move_2D(0.0, 0.0, 0.0)
                    log.info(f"Past the last {N_MOVING} moving obstacles: stopping")
                    raise SystemExit
            nav.in_zone = inside
            if now - nav.last_log > 1.0:
                nav.last_log = now
                goal = "none" if nav.goal is None else f"({nav.goal[0]:.2f}, {nav.goal[1]:.2f}){' final' if nav.goal_final else ''}"
                movers = [o for o in nav.tracks if o["moving"]] if nav.crossed else []
                info = "" if nav.crossed else " (not looked for until crossed)"
                if movers:
                    o = min(movers, key=lambda o: np.linalg.norm(o["p"] - c))
                    info = f" (nearest {np.linalg.norm(o['p'] - c):.2f} m, {2 * o['a']:.2f} m long, v ({o['v'][0]:+.2f}, {o['v'][1]:+.2f}))"
                    info += " at " + " ".join(f"({m['p'][0]:.2f}, {m['p'][1]:.2f}){'' if m['seen'] == m['t'] else ' hidden'}" for m in movers)
                log.info(f"pose ({c[0]:.2f}, {c[1]:.2f})  goal {goal}  {mode}  moving objects {len(movers)}{info}  crossed {nav.crossed}")

        # ---- main loop, every tick ----
        # Clock: in Gazebo the simulation's (from the odometry), so a simulation running slower than real time does not upset
        # its timing; on the real robot the computer's
        now = self.sim_time if self.is_simulation and self.sim_time is not None else time.monotonic()
        nav.now = now
        if nav.scan is None:
            return # wait for the first LiDAR scan

        # Sense: robot body centre in the world frame
        x, y, heading = self.pose
        h = math.radians(heading)
        c = np.array([x, y]) + nav.robot["body_offset"] * np.array([math.cos(h), math.sin(h)])
        if nav.origin is None: # first tick: centre the map on the start, and log the nearest object to check the LiDAR mounting
            nav.origin, nav.start = c - SIZE / 2, c.copy()
            r, a, valid = nav.scan
            i = int(np.argmin(np.where(valid, r, np.inf)))
            bearing = wrap_deg(math.degrees(a[i]) + nav.robot["lidar_yaw_deg"])
            log.info(f"Nearest object: {r[i]:.2f} m at bearing {bearing:+.0f} deg (0 = robot front, +90 = left)")

        # Map: add each new scan to the grid once, and track anything moving
        if nav.scan_seq != nav.used_seq:
            nav.used_seq = nav.scan_seq
            integrate(x, y, h, now)

        # Plan: goal = reachable frontier farthest from the start, path to it, twice a second
        if now - nav.last_plan > PLAN_PERIOD:
            nav.last_plan = now
            plan_path(c)

        # Act: follow the path facing the direction of travel, brake, time crossings, stop when done
        drive(c, h, now)
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