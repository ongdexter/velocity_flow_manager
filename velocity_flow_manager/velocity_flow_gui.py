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

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_srvs.srv import Trigger
from kr_tracker_msgs.srv import Transition


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

    def call_service(self, srv_name: str):
        # Create client and call; node/executor running in background thread will handle it
        client = self.node.create_client(Trigger, srv_name)
        # give a bit more time for services to appear
        self.node.get_logger().info(f'Attempting to call service "{srv_name}" from GUI (node ns={self.node.get_namespace()})')
        if not client.wait_for_service(timeout_sec=5.0):
            # log and notify user
            self.node.get_logger().warning(f'Service {srv_name} not available (wait timed out)')
            QtWidgets.QMessageBox.warning(self, 'Service missing', f'{srv_name} not available (timed out)')
            return
        req = Trigger.Request()
        try:
            fut = client.call_async(req)
        except Exception as e:
            self.node.get_logger().error(f'Failed to call service {srv_name}: {e}')
            QtWidgets.QMessageBox.warning(self, 'Service error', f'Failed to call {srv_name}: {e}')
            return

        # attach a callback
        def done_callback(future):
            try:
                res = future.result()
                # log and schedule GUI message in the Qt event loop (avoid calling Qt from rclpy thread)
                self.node.get_logger().info(f'Service {srv_name} response: success={res.success} message="{res.message}"')
                QtCore.QTimer.singleShot(0, lambda: QtWidgets.QMessageBox.information(self, 'Service response', str(res.success) + ': ' + res.message))
            except Exception as e:
                self.node.get_logger().warning(f'Service {srv_name} call failed: {e}')
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
        # Request the manager node to start publishing
        self.call_service('velocity_flow_manager/start')

    def on_stop(self):
        # Request the manager node to stop publishing
        self.call_service('velocity_flow_manager/stop')


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

def main(args=None):
    if not PYQT_AVAILABLE:
        print('PyQt5 not available. Install python3-pyqt5 or PyQt5 from pip to run the GUI.')
        return

    rclpy.init(args=args)
    node = Node('velocity_flow_gui')

    app = QtWidgets.QApplication(sys.argv)
    widget = GUIWidget(node)

    # start rclpy spinner thread
    spinner = RclpyThread(node)
    spinner.start()

    widget.show()
    try:
        rc = app.exec_()
    finally:
        # cleanup
        spinner.stop()
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(rc)


if __name__ == '__main__':
    main()
