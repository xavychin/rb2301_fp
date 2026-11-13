import argparse
import configparser
import heapq
import math
import os
import time
import types
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
        # Map-as-you-go navigation. Nothing about the course is assumed: the robot maps what its LiDAR sees, heads for the
        # reachable frontier farthest (along the path) from where it started, and times its crossing of anything it sees moving.
        RES = 0.04          # grid cell size [m]
        SIZE = 16.0         # grid side [m], centred on the start pose
        RANGE = 6.0         # LiDAR range used for mapping [m]
        PLAN_PERIOD = 0.5   # replanning period [s]
        LOOKAHEAD = 0.3     # pure pursuit lookahead [m]
        ZONE_MEMORY = 30.0  # how long a cell stays marked as "something moves here" [s]
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
            nav.zone = np.zeros((n, n), bool) # cells where something moved recently, grown by 2 cells
            nav.origin = nav.start = None
            nav.scan, nav.scan_seq, nav.used_seq = None, 0, 0
            nav.points = None # latest scan in the world frame: x, y, is_dynamic
            nav.tracks = [] # moving objects in the latest scan: dict(p, v, r, moving)
            nav.first_motion = None # time the first moving object was seen
            nav.path = nav.goal = None
            nav.goal_final = nav.narrow = False
            nav.last_plan = nav.last_log = 0.0
            nav.in_zone = nav.crossed = False
            nav.travel = np.zeros(2) # smoothed direction of travel
            nav.last_c = None
            nav.done_since = None

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
            '''Grow a boolean grid by `steps` cells (8-connected).'''
            for _ in range(steps):
                m = mask.copy()
                m[1:] |= mask[:-1]
                m[:-1] |= mask[1:]
                mask = m.copy()
                mask[:, 1:] |= m[:, :-1]
                mask[:, :-1] |= m[:, 1:]
            return mask

        def dijkstra(trav, start, goal=-1, cost=None):
            '''Shortest paths (in cells) over the True cells of `trav`, whose border must be False, from flat index `start`.
            Returns (dist, parent) as flat lists. Stops early once `goal` is reached.'''
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
            stat = hit & ~dyn
            miss[hi[hit], hj[hit]] = False
            nav.logodds[miss] -= 0.4
            nav.logodds[hi[stat], hj[stat]] += 0.85
            np.clip(nav.logodds, -2.0, 3.5, out=nav.logodds)
            nav.seen |= miss
            nav.seen[hi[hit], hj[hit]] = True
            nav.seen_time[nav.seen & np.isinf(nav.seen_time)] = now
            nav.points = (hx[valid], hy[valid], dyn[valid])
            track(hx[dyn], hy[dyn], hi[dyn], hj[dyn], now)

        def track(px, py, pi, pj, now):
            '''Group moving hits (consecutive beams) into objects and estimate their velocity from the previous scan.'''
            objects, start = [], 0
            for k in range(1, len(px) + 1):
                if k == len(px) or math.hypot(px[k] - px[k - 1], py[k] - py[k - 1]) > 0.2:
                    if k - start >= 2:
                        p = np.array([px[start:k].mean(), py[start:k].mean()])
                        rad = max(0.05, 0.5 * math.hypot(px[k - 1] - px[start], py[k - 1] - py[start]))
                        objects.append(dict(p=p, v=np.zeros(2), r=rad, moving=False, n=0, t=now, seen=now, cells=(pi[start:k], pj[start:k])))
                    start = k
            unmatched = list(nav.tracks)
            for o in objects:
                prev = [q for q in unmatched if np.linalg.norm(q["p"] - o["p"]) < 0.4]
                if prev:
                    q = min(prev, key=lambda q: np.linalg.norm(q["p"] - o["p"]))
                    unmatched = [u for u in unmatched if u is not q]
                    o["v"] = 0.5 * q["v"] + 0.5 * (o["p"] - q["p"]) / max(now - q["t"], 0.02)
                    o["n"] = q["n"] + (np.linalg.norm(o["v"]) > 0.1) # scans in a row seen moving
                    o["moving"] = q["moving"] or o["n"] >= 3
                if o["moving"]:
                    nav.dyn_time[o["cells"]] = now
                    nav.first_motion = nav.first_motion or now
            for q in unmatched: # keep moving objects that dropped out of view for a moment, at their predicted position
                if q["moving"] and now - q["seen"] < 1.0:
                    q["p"], q["t"] = q["p"] + q["v"] * (now - q["t"]), now
                    objects.append(q)
            nav.tracks = objects
            nav.zone = dilate((now - nav.dyn_time) < ZONE_MEMORY, 2)

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
            occ = (lo > 0.6) & ~((time.monotonic() - nav.dyn_time[i0:i1, j0:j1]) < ZONE_MEMORY)
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
            H, W = occ.shape
            local = lambda p: tuple(np.floor((p - nav.origin) / RES).astype(int) - (i0, j0))

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
                cand = frontier & trav & np.isfinite(ds) & (ds >= ds[rc] - 0.3 / RES) # never go back
                final = not cand.any()
                if final:
                    cand = trav & np.isfinite(ds)
                score = np.where(cand, ds, -1.0)
                g = np.unravel_index(int(np.argmax(score)), score.shape)
                if nav.goal is not None and not final and not nav.goal_final: # keep the goal unless clearly worse
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

        def crossing(c, ahead, now):
            '''Moving obstacles on the path ahead: ("clear",), ("wait", distance left to the stopping point) or ("dash",).'''
            hl = nav.robot["half_length"]
            cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(ahead, axis=0).T))]) + np.hypot(*(ahead[0] - c))
            keep = cum <= 1.5
            ahead, cum = ahead[keep], cum[keep]
            ci, cj = cell(ahead[:, 0], ahead[:, 1])
            inzone = nav.zone[np.clip(ci, 0, nav.n - 1), np.clip(cj, 0, nav.n - 1)]
            if not inzone.any():
                return ("clear",)
            s_in = cum[np.argmax(inzone)]
            if s_in <= hl:
                return ("dash",) # already in a zone: never stop inside it
            s_out, last = s_in, s_in
            for s, z in zip(cum, inzone): # end of the zone, treating gaps too short to wait in as part of it
                if z and s - last < 2 * hl + 0.2:
                    s_out = s
                if z:
                    last = s
            stop = s_in - hl - 0.08
            if now - nav.first_motion < 2.0:
                return ("wait", stop) # watch for a moment first
            v = 0.75 * nav.dash # assume the robot is slower than commanded (turning, braking)
            ts = np.arange(0.0, (s_out + hl) / v + 0.3, 0.1)
            q = np.stack([np.interp(v * ts, cum, ahead[:, 0]), np.interp(v * ts, cum, ahead[:, 1])], 1)
            safe_r = math.hypot(hl, nav.robot["half_width"]) + 0.1
            for o in nav.tracks:
                if o["moving"]:
                    speeds = (1.0, 0.5) if np.linalg.norm(o["v"]) > 0.1 else (0.0,) # slowing down counts too
                    for f in speeds:
                        if np.min(np.linalg.norm(o["p"] + ts[:, None] * f * o["v"] - q, axis=1)) < o["r"] + safe_r:
                            return ("wait", stop)
            return ("dash",)

        def brake_factor(c, h, direction, dashing):
            '''Speed factor in [0, 1] from the raw scan in the direction of motion (moving things ignored while dashing),
            and the offset of the closest point in the way.'''
            px, py, dyn = nav.points
            keep = np.ones(len(px), bool)
            if dashing: # never stop inside a crossing because of the moving things themselves
                pi, pj = cell(px, py)
                keep = ~dyn & ~nav.zone[np.clip(pi, 0, nav.n - 1), np.clip(pj, 0, nav.n - 1)]
            rx, ry = px[keep] - c[0], py[keep] - c[1]
            par = direction[0] * rx + direction[1] * ry
            perp = np.abs(direction[0] * ry - direction[1] * rx)
            fwd = abs(direction[0] * math.cos(h) + direction[1] * math.sin(h))
            hl, hw = nav.robot["half_length"], nav.robot["half_width"]
            along, across = hl * fwd + hw * (1 - fwd), hw * fwd + hl * (1 - fwd)
            sel = np.nonzero((par > 0) & (perp < across + 0.01))[0]
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
            mode = crossing(c, P[i:], now) if nav.first_motion else ("clear",)
            if mode[0] == "clear": # look before entering space just seen, so anything moving there shows itself
                near = P[i:][d[i:] < 0.6 + nav.robot["half_length"]]
                pi, pj = cell(near[:, 0], near[:, 1])
                fresh = now - nav.seen_time[np.clip(pi, 0, nav.n - 1), np.clip(pj, 0, nav.n - 1)] < 1.5
                if fresh.any():
                    mode = ("wait", d[i:][d[i:] < 0.6 + nav.robot["half_length"]][np.argmax(fresh)] - nav.robot["half_length"] - 0.05)
            if mode[0] == "wait":
                speed = min(speed, max(0.0, 1.5 * mode[1]))
            elif mode[0] == "dash":
                speed = nav.dash
            if norm > 1e-3:
                brake, block = brake_factor(c, h, direction, mode[0] == "dash")
                if brake < 0.2: # blocked: slide along the obstacle instead of pushing into it
                    away = block / np.linalg.norm(block)
                    slide = direction - (direction @ away) * away
                    if np.linalg.norm(slide) > 0.3:
                        direction = slide / np.linalg.norm(slide)
                        brake = 0.5 * brake_factor(c, h, direction, mode[0] == "dash")[0]
                speed *= brake
            err = wrap_deg(math.degrees(math.atan2(direction[1], direction[0]) - h)) if norm > 1e-3 else 0.0
            turn = float(np.clip(2.0 * math.radians(err), -nav.turn_max, nav.turn_max)) # face the direction of travel
            if abs(err) > 45 and mode[0] != "dash":
                speed *= 0.5
            vx, vy = direction * speed
            bo = nav.robot["body_offset"] # command base_footprint so the body centre moves as asked
            vx, vy = vx + turn * bo * math.sin(h), vy - turn * bo * math.cos(h)
            self.move_2D(math.cos(h) * vx + math.sin(h) * vy, -math.sin(h) * vx + math.cos(h) * vy, turn)
            check_finish(c, now, mode[0])

        def check_finish(c, now, mode):
            '''Done once the robot has crossed a moving-obstacle zone, nothing moves ahead and nowhere is left to go.'''
            if nav.last_c is not None and np.linalg.norm(c - nav.last_c) > 0.02:
                nav.travel = 0.8 * nav.travel + 0.2 * (c - nav.last_c) / np.linalg.norm(c - nav.last_c)
                nav.last_c = c
            elif nav.last_c is None:
                nav.last_c = c
            ci, cj = cell(c[0], c[1])
            k = int(nav.robot["half_length"] / RES)
            inside = bool(nav.zone[max(ci - k, 0):ci + k + 1, max(cj - k, 0):cj + k + 1].any())
            if nav.in_zone and not inside:
                nav.crossed = True
            nav.in_zone = inside
            zi, zj = np.nonzero(nav.zone[max(ci - 38, 0):ci + 39, max(cj - 38, 0):cj + 39])
            rel = (np.stack([zi + max(ci - 38, 0), zj + max(cj - 38, 0)], 1) + 0.5) * RES + nav.origin - c
            ahead = bool(np.any((rel @ nav.travel > 0.1) & (np.hypot(rel[:, 0], rel[:, 1]) < 1.5)))
            if now - nav.last_log > 1.0:
                nav.last_log = now
                goal = "none" if nav.goal is None else f"({nav.goal[0]:.2f}, {nav.goal[1]:.2f}){' final' if nav.goal_final else ''}"
                log.info(f"pose ({c[0]:.2f}, {c[1]:.2f})  goal {goal}  {mode}  moving objects {sum(o['moving'] for o in nav.tracks)}  crossed {nav.crossed}")
            if nav.crossed and nav.goal_final and not ahead and math.dist(c, nav.goal) < 0.25:
                nav.done_since = nav.done_since or now
                if now - nav.done_since > 1.0:
                    self.move_2D(0.0, 0.0, 0.0)
                    log.info("Past the moving obstacles: done")
                    raise SystemExit
            else:
                nav.done_since = None

        # ---- main loop, every tick ----
        now = time.monotonic()
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