#!/usr/bin/env python3
import sys
import os
import select
import termios
import time
import tty
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# Control usage explanation
msg = """
===================================================
Custom Drone Teleop Keyboard Controller
===================================================
Control Your Drone Using:
  - Arrow Keys to Move:
        Up Arrow: Move Forward
      Down Arrow: Move Backward
      Left Arrow: Move Left
     Right Arrow: Move Right

  - Altitude Control:
               w: Move Up
               s: Move Down

  - Yaw Control (Rotation):
               a: Rotate Left (Yaw CCW)
               d: Rotate Right (Yaw CW)

  - Stop Control:
               k: Hover / Stop all movement

  You can hold down multiple keys together (e.g. Up Arrow + a
  to move forward and rotate at the same time) - movements are added together.

Press Ctrl+C to quit.
===================================================
"""

# Key mappings
UP_ARROW = '\x1b[A'
DOWN_ARROW = '\x1b[B'
RIGHT_ARROW = '\x1b[C'
LEFT_ARROW = '\x1b[D'

moveBindings = {
    UP_ARROW: (0.5, 0.0, 0.0, 0.0),    # (linear x, linear y, linear z, angular z)
    DOWN_ARROW: (-0.5, 0.0, 0.0, 0.0),
    LEFT_ARROW: (0.0, 0.5, 0.0, 0.0),
    RIGHT_ARROW: (0.0, -0.5, 0.0, 0.0),
    'w': (0.0, 0.0, 0.5, 0.0),
    's': (0.0, 0.0, -0.5, 0.0),
    'a': (0.0, 0.0, 0.0, 1.0),
    'd': (0.0, 0.0, 0.0, -1.0),
}
STOP_KEYS = ('k', ' ')

# Maximum time a key is considered "still pressed". The terminal sends a held
# key repeatedly at the OS repeat rate (typically ~30-50ms);
# if no new repeat arrives within this window, the key is considered released.
# Because there is no real "key released" event in terminal input, this is the
# only way to simulate multiple keys being held down at the same time.
KEY_HOLD_TIMEOUT = 0.3


def read_available_keys(fd):
    """Read ALL keys waiting in stdin during this loop iteration.

    IMPORTANT: raw terminal mode is NOT set here, only once at the beginning
    of the program (see TeleopNode.run). tty.setraw() by default uses
    TCSAFLUSH - meaning it DELETES all pending input that has not been READ
    every time it's called. Calling this in every loop iteration (as it was
    done here previously), randomly deleted the data arriving in the few
    milliseconds window between two keystrokes, causing keys to be lost/
    fragmented - this was the root cause of the arrow key and multi-key issue.

    select() works at the file descriptor (fd) level, but sys.stdin.read()
    keeps its own buffer in Python - mixing the two was another source of
    inconsistency. Instead, we use os.read() to directly read ALL available
    bytes from the fd at once, and parse them in our own memory.
    """
    rlist, _, _ = select.select([fd], [], [], 0.05)
    if not rlist:
        return []

    chunk = os.read(fd, 1024)
    # If the read finishes before the escape sequence is complete (e.g. only \x1b or \x1b[)
    # make a short additional attempt to complete the remaining 1-2 bytes.
    for _ in range(2):
        if chunk[-2:] in (b'\x1b', b'\x1b[') or chunk[-1:] == b'\x1b':
            rlist2, _, _ = select.select([fd], [], [], 0.05)
            if not rlist2:
                break
            chunk += os.read(fd, 8)
        else:
            break

    text = chunk.decode(errors='ignore')
    keys = []
    i = 0
    while i < len(text):
        if text[i] == '\x1b' and i + 2 < len(text):
            keys.append(text[i:i + 3])
            i += 3
        else:
            keys.append(text[i])
            i += 1
    return keys


class TeleopNode(Node):
    def __init__(self):
        super().__init__('drone_teleop')
        self.publisher_ = self.create_publisher(Twist, '/x500/cmd_vel', 10)
        self.fd = sys.stdin.fileno()
        self.settings = termios.tcgetattr(self.fd)
        self.get_logger().info("Drone Teleop Node started successfully.")

    def run(self):
        active_keys = {}  # key -> time last seen (time.monotonic())
        tty.setraw(self.fd)  # raw mode is set ONLY here, once
        try:
            print(msg)
            while rclpy.ok():
                now = time.monotonic()
                for key in read_available_keys(self.fd):
                    if key == '\x03':  # Ctrl+C
                        return
                    if key in STOP_KEYS:
                        active_keys.clear()
                    elif key in moveBindings:
                        active_keys[key] = now

                # Discard keys that did not arrive again this round (no longer pressed)
                active_keys = {
                    k: t for k, t in active_keys.items()
                    if now - t <= KEY_HOLD_TIMEOUT
                }

                x = y = z = th = 0.0
                for dx, dy, dz, dth in (moveBindings[k] for k in active_keys):
                    x += dx
                    y += dy
                    z += dz
                    th += dth

                if active_keys:
                    print(f"\rActive: {','.join(active_keys):<20} "
                          f"(vx:{x:.1f} vy:{y:.1f} vz:{z:.1f} yaw:{th:.1f})   ", end="")
                else:
                    print("\rHover / Stop" + " " * 40, end="")

                twist = Twist()
                twist.linear.x = float(x)
                twist.linear.y = float(y)
                twist.linear.z = float(z)
                twist.angular.z = float(th)
                self.publisher_.publish(twist)
        except Exception as e:
            print(e)
        finally:
            # Publish stop command on shutdown
            self.publisher_.publish(Twist())
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
