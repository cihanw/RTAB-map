#!/usr/bin/env python3
"""Terminal chat client for the LLM bridge.

Deliberately a separate process from the bridge: the bridge must keep flying
the drone whether or not anyone is watching a terminal, and a console that
crashes (or a closed SSH session) must not take the commander down with it.
"""

import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ANSI colours, dropped automatically when output is piped to a file.
_TTY = sys.stdout.isatty()
BOLD = '\033[1m' if _TTY else ''
DIM = '\033[2m' if _TTY else ''
CYAN = '\033[36m' if _TTY else ''
YELLOW = '\033[33m' if _TTY else ''
RESET = '\033[0m' if _TTY else ''

BANNER = f"""{BOLD}Drone LLM console{RESET}
Type a command in plain English, or {BOLD}quit{RESET} to exit.
  examples: "explore the area" | "fly to 3, -2" | "draw a star with radius 3"
            "stop" | "how are you doing?"
{DIM}Telemetry narration appears automatically every ~5s.{RESET}
"""

PROMPT = f'{BOLD}you>{RESET} '


class Console(Node):

    def __init__(self):
        super().__init__('llm_console')
        self.pub = self.create_publisher(String, '/llm/user_input', 10)
        self.create_subscription(String, '/llm/response', self._on_response, 10)
        self.create_subscription(
            String, '/llm/narration', self._on_narration, 10)

    def _write(self, text):
        # Redraw the prompt after async output so the input line is not eaten.
        sys.stdout.write(f'\r\033[K{text}\n{PROMPT}')
        sys.stdout.flush()

    def _on_response(self, msg):
        self._write(f'{CYAN}drone>{RESET} {msg.data}')

    def _on_narration(self, msg):
        self._write(f'{YELLOW}[telemetry]{RESET} {DIM}{msg.data}{RESET}')

    def send(self, text):
        self.pub.publish(String(data=text))


def main(args=None):
    rclpy.init(args=args)
    node = Console()

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print(BANNER)
    try:
        while rclpy.ok():
            try:
                text = input(PROMPT).strip()
            except EOFError:
                break
            if not text:
                continue
            if text.lower() in ('quit', 'exit'):
                break
            node.send(text)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
