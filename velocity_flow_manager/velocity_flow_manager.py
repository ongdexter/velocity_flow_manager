"""Velocity Flow Manager node

This node subscribes to /flow/raw (Float32MultiArray), computes desired
velocity commands (placeholder), and publishes VelocityGoal messages to the
configured topic. It exposes simple Trigger services to start/stop
publishing so the GUI can control its state.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger
from kr_tracker_msgs.msg import VelocityGoal

import numpy as np
import cv2


def reshape_multiarray(msg: Float32MultiArray):
    # Expect layout dim ordering [channel, height, width]
    if not (msg.layout and msg.layout.dim and len(msg.layout.dim) >= 3):
        # fallback: return flat numpy array
        return np.array(msg.data)

    c = msg.layout.dim[0].size
    h = msg.layout.dim[1].size
    w = msg.layout.dim[2].size
    arr = np.array(msg.data)
    try:
        arr = arr.reshape((c, h, w))
    except Exception:
        return np.array(msg.data)
    return arr


def compute_desired_velocity(flow):
    """Compute desired (vx, vy, vz) from flow array.

    flow is expected as a numpy array with shape (2, H, W) where flow[0] is u
    and flow[1] is v.
    Returns vx, vy, vz (vyaw is handled as 0.0 by publisher).
    """
    if flow is None:
        return 0.0, 0.0, 0.0

    mag_filter = 2.5  # pixels

    # defensive: ensure shape
    if flow.ndim != 3 or flow.shape[0] < 2:
        return 0.0, 0.0, 0.0

    flow_width = flow.shape[2]
    flow_height = flow.shape[1]
    u = flow[0, :, :]
    v = flow[1, :, :]
    u = u * flow_width
    v = v * flow_height

    magnitude = np.sqrt(u ** 2 + v ** 2)
    magnitude[magnitude < mag_filter] = 0.0

    if magnitude.sum() < 10:
        return 0.0, 0.0, 0.0

    mask = (magnitude >= mag_filter).astype(np.uint8)

    clusters = []
    if mask.sum() > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for lbl in range(1, num_labels):
            area = int(stats[lbl, 4])
            if area < 10:
                continue
            left = int(stats[lbl, 0])
            top = int(stats[lbl, 1])
            width_box = int(stats[lbl, 2])
            height_box = int(stats[lbl, 3])
            cx, cy = centroids[lbl]

            mask_lbl = (labels == lbl)
            if np.count_nonzero(mask_lbl) == 0:
                continue
            avg_u_comp = np.mean(u[mask_lbl])
            avg_v_comp = np.mean(v[mask_lbl])

            clusters.append({
                'label': lbl,
                'area': area,
                'bbox': (left, top, width_box, height_box),
                'centroid': (cx, cy),
                'avg_u': avg_u_comp,
                'avg_v': avg_v_comp,
            })

    velocity_vector = (0.0, 0.0, 0.0)
    if len(clusters) > 0:
        cluster = clusters[0]
        centroid = cluster['centroid']
        if centroid[0] < flow_width / 2:
            velocity_vector = (2.0, 0.0, 0.0)
        else:
            velocity_vector = (-2.0, 0.0, 0.0)

    return velocity_vector


class VelocityFlowManager(Node):
    def __init__(self):
        super().__init__('velocity_flow_manager')

        self.declare_parameter('publish_topic', '/velocity_goal')
        publish_topic = self.get_parameter('publish_topic').value
        self.pub = self.create_publisher(VelocityGoal, publish_topic, 10)

        # internal state
        self.publishing = False
        self.latest_flow = None
        self.executing_dodge = False
        self.executed_once = False
        self.execution_velocity = None
        self.execution_start_time = None
        self.execution_duration = 1.0

        # subscription
        self.create_subscription(Float32MultiArray, '/flow/raw', self._flow_cb, 10)

        # services to control publishing
        self.create_service(Trigger, 'velocity_flow_manager/start', self._on_start)
        self.create_service(Trigger, 'velocity_flow_manager/stop', self._on_stop)

        # periodic publisher
        self.create_timer(0.1, self._timer_cb)

        self.get_logger().info('Velocity Flow Manager node initialized')

    def _flow_cb(self, msg: Float32MultiArray):
        if self.publishing is False:
            return
        try:
            arr = reshape_multiarray(msg)
            self.latest_flow = arr
        except Exception:
            self.get_logger().warning('Failed to reshape incoming Float32MultiArray')
            self.latest_flow = None

        # trigger dodge behaviour once when a significant flow is detected
        if not self.executing_dodge and not self.executed_once:
            vx, vy, vz = compute_desired_velocity(self.latest_flow)
            if abs(vx) > 0.0 or abs(vy) > 0.0 or abs(vz) > 0.0:
                self.executing_dodge = True
                self.execution_velocity = (vx, vy, vz)
                self.execution_start_time = self.get_clock().now()
                self.get_logger().info(f'Starting dodge maneuver with velocity ({vx}, {vy}, {vz})')

    def _on_start(self, request, response):
        self.publishing = True
        response.success = True
        response.message = 'started'
        self.get_logger().info('Velocity-flow controller started')
        return response

    def _on_stop(self, request, response):
        self.publishing = False
        response.success = True
        response.message = 'stopped'
        self.get_logger().info('Velocity-flow controller stopped')
        return response

    def _timer_cb(self):
        msg = VelocityGoal()
        if self.publishing:
            if self.executing_dodge:
                vx, vy, vz = self.execution_velocity
                msg.vx = float(vx)
                msg.vy = float(vy)
                msg.vz = float(vz)
                msg.vyaw = 0.0

                curr_time = self.get_clock().now()
                elapsed = (curr_time - self.execution_start_time).nanoseconds / 1e9
                if elapsed >= self.execution_duration:
                    # finished dodge
                    self.executing_dodge = False
                    self.executed_once = True
                    self.execution_velocity = None
                    self.execution_start_time = None
                    msg.vx = 0.0
                    msg.vy = 0.0
                    msg.vz = 0.0
                    msg.vyaw = 0.0
                    self.get_logger().info('Finished dodge maneuver')
                else:
                    # continue executing dodge
                    vx, vy, vz = self.execution_velocity
                    msg.vx = float(vx)
                    msg.vy = float(vy)
                    msg.vz = float(vz)
                    msg.vyaw = 0.0
            else:
                msg.vx = 0.0
                msg.vy = 0.0
                msg.vz = 0.0
                msg.vyaw = 0.0
        else:
            msg.vx = 0.0
            msg.vy = 0.0
            msg.vz = 0.0
            msg.vyaw = 0.0
        msg.use_position_gains = False
        try:
            self.pub.publish(msg)
        except Exception:
            self.get_logger().warning('Publish failed')


def main(args=None):
    rclpy.init(args=args)
    node = VelocityFlowManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
