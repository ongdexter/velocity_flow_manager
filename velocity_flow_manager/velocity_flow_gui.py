"""Simple PyQt5 + rclpy GUI to call Initialize / Start / Stop Trigger services

This is intentionally minimal. It starts an rclpy node and executor in a
background thread and offers three buttons which call Trigger services.

If PyQt5 is not available the module will print a short message instead of
launching the UI.
"""

import sys
import threading

from PyQt5 import QtWidgets, QtCore
PYQT_AVAILABLE = True

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_srvs.srv import Trigger
from kr_tracker_msgs.srv import Transition
from kr_tracker_msgs.msg import VelocityGoal
from std_msgs.msg import Float32MultiArray


class GUIWidget(QtWidgets.QWidget):
    def __init__(self, node: Node):
        super().__init__()
        self.node = node
        self.setWindowTitle('Velocity Manager GUI')
        self.resize(320, 120)

        layout = QtWidgets.QVBoxLayout()

        self.btn_init = QtWidgets.QPushButton('Initialize')
        self.btn_start = QtWidgets.QPushButton('Update')
        self.btn_stop = QtWidgets.QPushButton('Stop')

        layout.addWidget(self.btn_init)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)

        # velocity inputs
        form = QtWidgets.QFormLayout()
        self.spin_vx = QtWidgets.QDoubleSpinBox()
        self.spin_vx.setRange(-5.0, 5.0)
        self.spin_vx.setSingleStep(0.1)
        self.spin_vx.setValue(0.0)

        self.spin_vy = QtWidgets.QDoubleSpinBox()
        self.spin_vy.setRange(-5.0, 5.0)
        self.spin_vy.setSingleStep(0.1)
        self.spin_vy.setValue(0.0)

        self.spin_vz = QtWidgets.QDoubleSpinBox()
        self.spin_vz.setRange(-5.0, 5.0)
        self.spin_vz.setSingleStep(0.1)
        self.spin_vz.setValue(0.0)

        self.spin_vyaw = QtWidgets.QDoubleSpinBox()
        self.spin_vyaw.setRange(-10.0, 10.0)
        self.spin_vyaw.setSingleStep(0.1)
        self.spin_vyaw.setValue(0.0)

        form.addRow('vx (m/s):', self.spin_vx)
        form.addRow('vy (m/s):', self.spin_vy)
        form.addRow('vz (m/s):', self.spin_vz)
        form.addRow('vyaw (rad/s):', self.spin_vyaw)

        layout.addLayout(form)

        self.setLayout(layout)

        self.btn_init.clicked.connect(lambda: self.call_transition('VelocityTracker', 'trackers_manager/transition'))
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)

        # internal publishing state
        # publishing flag controls whether timer publishes computed velocity
        self.publishing = False
        # latest flow data received on /flow/raw (2D array as nested lists) or None
        self.latest_flow = None

        self.executing_dodge = False
        self.executed_once = False
        self.execution_velocity = None
        self.execution_start_time = None
        self.execution_duration = 1.0

        # publish topic (configurable if needed)
        self.publish_topic = '/velocity_goal'

    def call_service(self, srv_name: str):
        # Create client and call; node/executor running in background thread will handle it
        client = self.node.create_client(Trigger, srv_name)
        if not client.wait_for_service(timeout_sec=1.0):
            QtWidgets.QMessageBox.warning(self, 'Service missing', f'{srv_name} not available')
            return
        req = Trigger.Request()
        fut = client.call_async(req)

        # attach a callback
        def done_callback(future):
            try:
                res = future.result()
                # schedule GUI message in the Qt event loop (avoid calling Qt from rclpy thread)
                QtCore.QTimer.singleShot(0, lambda: QtWidgets.QMessageBox.information(self, 'Service response', str(res.success) + ': ' + res.message))
            except Exception as e:
                QtCore.QTimer.singleShot(0, lambda: QtWidgets.QMessageBox.warning(self, 'Service error', str(e)))

        fut.add_done_callback(done_callback)

    def call_transition(self, tracker_name: str, srv_name: str):
        """Call the Transition service on the trackers_manager to switch active tracker.

        tracker_name: e.g. 'kr_trackers/VelocityTracker'
        srv_name: service name relative to this node's namespace, e.g. 'trackers_manager/transition'
        """
        client = self.node.create_client(Transition, srv_name)
        if not client.wait_for_service(timeout_sec=2.0):
            QtWidgets.QMessageBox.warning(self, 'Service missing', f'{srv_name} not available')
            return
        req = Transition.Request()
        req.tracker = tracker_name
        fut = client.call_async(req)

        def done(future):
            try:
                res = future.result()
                QtCore.QTimer.singleShot(0, lambda: QtWidgets.QMessageBox.information(self, 'Transition response', str(res.success) + ': ' + res.message))
            except Exception as e:
                QtCore.QTimer.singleShot(0, lambda: QtWidgets.QMessageBox.warning(self, 'Service error', str(e)))

        fut.add_done_callback(done)

    def on_start(self):
        # enable publishing; velocities will be computed from subscribed flow data
        self.publishing = True
        QtWidgets.QMessageBox.information(self, 'Started', f'Started publishing to {self.publish_topic}')

    def on_stop(self):
        # stop publishing (the background publisher will publish zeros)
        self.publishing = False
        QtWidgets.QMessageBox.information(self, 'Stopped', f'Stopped publishing to {self.publish_topic}')


class RclpyThread(threading.Thread):
    def __init__(self, node: Node):
        super().__init__(daemon=True)
        self.node = node
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self._running = True

    def run(self):
        while self._running:
            self.executor.spin_once(timeout_sec=0.1)

    def stop(self):
        self._running = False
        # give a moment for spin loop to exit
        self.executor.remove_node(self.node)

def reshape_multiarray(msg: Float32MultiArray):
    c, h, w = msg.layout.dim[0].size, msg.layout.dim[1].size, msg.layout.dim[2].size
    flow = msg.data

    flow = np.array(flow).reshape(c, h, w)

    return flow


def compute_desired_velocity(flow):
    """Placeholder: compute desired (vx, vy, vz, vyaw) from flow array.

    flow: nested list or None. For now return zeros; user will implement.
    """
    if flow is None:
        return 0.0, 0.0, 0.0
    
    mag_filter = 3 # pixels
    
    flow_width = flow.shape[2]
    flow_height = flow.shape[1]
    u = flow[0, :, :]  # horizontal flow
    v = flow[1, :, :]  # vertical flow
    u *= flow_width
    v *= flow_height

    magnitude = np.sqrt(u**2 + v**2)
    magnitude[magnitude < mag_filter] = 0.0

    if magnitude.sum() < 10:
        return 0.0, 0.0, 0.0
    
    # cluster flow vectors, should be at least 50 pixels, run connectedcomponents
    # create a binary mask of significant motion
    mask = (magnitude >= mag_filter).astype(np.uint8)    

    # connected components on mask
    clusters = []
    if mask.sum() > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # stats columns: [left, top, width, height, area]
        for lbl in range(1, num_labels):
            area = int(stats[lbl, 4])
            if area < 50:
                continue
            left = int(stats[lbl, 0])
            top = int(stats[lbl, 1])
            width_box = int(stats[lbl, 2])
            height_box = int(stats[lbl, 3])
            cx, cy = centroids[lbl]

            # compute average vector inside this cluster
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
    # average flow vectors in biggest cluster
    if len(clusters) > 0:
        cluster = clusters[0]
        centroid = cluster['centroid']
        if centroid[0] < flow_width / 2:            
            # send velocity vector to right (in vicon coordinate frame)
            velocity_vector = np.array([0.5, 0.0, 0.0])  # vx, vy, vz
        else:
            velocity_vector = np.array([-0.5, 0.0, 0.0])  # vx, vy, vz
    
    return velocity_vector


def main(args=None):
    if not PYQT_AVAILABLE:
        print('PyQt5 not available. Install python3-pyqt5 or PyQt5 from pip to run the GUI.')
        return

    rclpy.init(args=args)
    node = Node('velocity_manager_node')

    app = QtWidgets.QApplication(sys.argv)
    widget = GUIWidget(node)

    # start rclpy spinner thread
    spinner = RclpyThread(node)
    spinner.start()

    # declare/read parameter for publish topic (similar to waypoint_manager)
    publish_topic_param = node.declare_parameter('publish_topic', widget.publish_topic)
    publish_topic = publish_topic_param.value
    pub = node.create_publisher(VelocityGoal, publish_topic, 10)

    # subscribe to flow raw data and store latest into widget.latest_flow
    def flow_cb(msg: Float32MultiArray):
        # node.get_logger().info('Received flow raw message')
        arr = reshape_multiarray(msg) # (2, 256, 320)
        # store raw flow array (do not touch Qt widgets from this thread)
        widget.latest_flow = arr

        # only trigger this once
        if not widget.executing_dodge and not widget.executed_once:
            vx, vy, vz = compute_desired_velocity(widget.latest_flow)
            if abs(vx) > 0.0 or abs(vy) > 0.0 or abs(vz) > 0.0:
                widget.executing_dodge = True
                widget.execution_velocity = (vx, vy, vz)
                curr_time = node.get_clock().now()
                widget.execution_start_time = curr_time
                node.get_logger().info(f'Starting dodge maneuver with velocity ({vx}, {vy}, {vz})')

    sub = node.create_subscription(Float32MultiArray, '/flow/raw', flow_cb, 10)

    def timer_cb():
        # timer runs in the node's executor thread, safe to access widget attrs
        msg = VelocityGoal()
        if widget.publishing:
            if widget.executing_dodge:
                vx, vy, vz = widget.execution_velocity
                msg.vx = float(vx)
                msg.vy = float(vy)
                msg.vz = float(vz)
                msg.vyaw = 0.0

                curr_time = node.get_clock().now()
                elapsed = (curr_time - widget.execution_start_time).nanoseconds / 1e9
                if elapsed >= widget.execution_duration:
                    # finished dodge
                    widget.executing_dodge = False
                    widget.executed_once = True
                    widget.execution_velocity = None
                    widget.execution_start_time = None
                    msg.vx = 0.0
                    msg.vy = 0.0
                    msg.vz = 0.0
                    msg.vyaw = 0.0
                    node.get_logger().info('Finished dodge maneuver')                
                else:
                    # continue executing dodge
                    vx, vy, vz = widget.execution_velocity
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
            # node.get_logger().info('Not updating, sending zeros')
            msg.vx = 0.0
            msg.vy = 0.0
            msg.vz = 0.0
            msg.vyaw = 0.0
        msg.use_position_gains = False
        try:
            pub.publish(msg)
        except Exception:
            pass

    timer_period = 0.02
    timer = node.create_timer(timer_period, timer_cb)

    widget.show()
    try:
        rc = app.exec_()
    finally:
        # cleanup
        spinner.stop()
        # cancel timer and let node destroy it
        try:
            timer.cancel()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(rc)


if __name__ == '__main__':
    main()
