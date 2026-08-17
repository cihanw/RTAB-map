#!/usr/bin/env python3
"""Browser UI for the LLM bridge.

Same role as console.py - a client sitting on the bridge's public interface
(/llm/user_input, /llm/response, /llm/narration) - just with an HTML front end
instead of a terminal. Deliberately stdlib-only (http.server): flask/fastapi
are not installed in this container, and pulling one in for a single page
would be the third dependency-install detour of this project.
"""

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

HOST = '0.0.0.0'
PORT = 8080

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Drone LLM Commander</title>
<style>
  :root { color-scheme: dark; }
  body {
    margin: 0; height: 100vh; display: flex; flex-direction: column;
    background: #12151a; color: #d8dee9;
    font: 14px/1.5 ui-monospace, "Cascadia Code", "Fira Code", monospace;
  }
  header {
    padding: 10px 16px; border-bottom: 1px solid #2a2f3a;
    display: flex; align-items: center; gap: 10px;
  }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; color: #f0f2f5; }
  #dot { width: 9px; height: 9px; border-radius: 50%; background: #555; }
  #dot.up { background: #4caf72; }
  #dot.down { background: #d9534f; }
  #log {
    flex: 1; overflow-y: auto; padding: 14px 16px;
    display: flex; flex-direction: column; gap: 6px;
  }
  .row { white-space: pre-wrap; word-break: break-word; }
  .you   { color: #d8dee9; }
  .you .tag   { color: #d8dee9; font-weight: 700; }
  .drone .tag { color: #56b6c2; font-weight: 700; }
  .tele  { color: #7a828e; font-style: italic; }
  .tele .tag  { color: #c9a227; font-style: normal; font-weight: 700; }
  form {
    display: flex; gap: 8px; padding: 12px 16px;
    border-top: 1px solid #2a2f3a; background: #171b22;
  }
  input {
    flex: 1; background: #0d0f13; color: #e8eaed; border: 1px solid #2a2f3a;
    border-radius: 6px; padding: 10px 12px; font: inherit;
  }
  input:focus { outline: none; border-color: #56b6c2; }
  button {
    background: #56b6c2; color: #0d0f13; border: none; border-radius: 6px;
    padding: 0 18px; font: inherit; font-weight: 700; cursor: pointer;
  }
  button:disabled { opacity: .5; cursor: default; }
</style>
</head>
<body>
<header><span id="dot"></span><h1>Drone LLM Commander</h1></header>
<div id="log"></div>
<form id="f">
  <input id="input" autocomplete="off" placeholder="explore the area / fly to 2, -1 / draw a star / stop" autofocus>
  <button id="send">Send</button>
</form>
<script>
const log = document.getElementById('log');
const dot = document.getElementById('dot');
const form = document.getElementById('f');
const input = document.getElementById('input');

function addRow(cls, tag, text) {
  const row = document.createElement('div');
  row.className = 'row ' + cls;
  const t = document.createElement('span');
  t.className = 'tag';
  t.textContent = tag + ' ';
  row.appendChild(t);
  row.appendChild(document.createTextNode(text));
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}

function connect() {
  const es = new EventSource('/api/events');
  es.onopen = () => dot.className = 'up';
  es.onerror = () => { dot.className = 'down'; es.close(); setTimeout(connect, 2000); };
  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.kind === 'response') addRow('drone', 'drone>', msg.text);
    else if (msg.kind === 'narration') addRow('tele', '[telemetry]', msg.text);
  };
}
connect();

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  addRow('you', 'you>', text);
  input.value = '';
  fetch('/api/command', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text}),
  }).catch(() => addRow('tele', '[ui]', 'Failed to send - is the bridge running?'));
});
</script>
</body>
</html>
"""


class Hub:
    """Fan-out of ROS messages to every connected browser tab's SSE stream."""

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers = set()

    def subscribe(self):
        q = queue.Queue(maxsize=64)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subscribers.discard(q)

    def publish(self, kind, text):
        payload = json.dumps({'kind': kind, 'text': text})
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass   # a stalled tab should not block live narration for others


class BridgeClient(Node):
    """The ROS side: publishes commands, forwards responses/narration to the Hub."""

    def __init__(self, hub):
        super().__init__('llm_web_ui')
        self.hub = hub
        self.pub = self.create_publisher(String, '/llm/user_input', 10)
        self.create_subscription(
            String, '/llm/response',
            lambda m: hub.publish('response', m.data), 10)
        self.create_subscription(
            String, '/llm/narration',
            lambda m: hub.publish('narration', m.data), 10)

    def send(self, text):
        self.pub.publish(String(data=text))


def make_handler(hub):

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, fmt, *args):
            pass   # keep stdout to what the ROS node itself logs

        def do_GET(self):
            if self.path == '/':
                body = PAGE.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == '/api/events':
                self._stream_events()
                return
            self.send_error(404)

        def do_POST(self):
            if self.path != '/api/command':
                self.send_error(404)
                return
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length) if length else b'{}'
            try:
                text = json.loads(raw).get('text', '').strip()
            except json.JSONDecodeError:
                text = ''
            if text:
                self.server.bridge.send(text)
            self.send_response(202)
            self.send_header('Content-Length', '0')
            self.end_headers()

        def _stream_events(self):
            q = hub.subscribe()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            try:
                while True:
                    try:
                        payload = q.get(timeout=15.0)
                        self.wfile.write(f'data: {payload}\n\n'.encode('utf-8'))
                    except queue.Empty:
                        self.wfile.write(b': keepalive\n\n')   # SSE comment
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass   # browser tab closed/navigated away
            finally:
                hub.unsubscribe(q)

    return Handler


def main(args=None):
    rclpy.init(args=args)
    hub = Hub()
    bridge = BridgeClient(hub)

    threading.Thread(target=rclpy.spin, args=(bridge,), daemon=True).start()

    server = ThreadingHTTPServer((HOST, PORT), make_handler(hub))
    server.bridge = bridge
    server.daemon_threads = True

    bridge.get_logger().info(f'Web UI listening on http://{HOST}:{PORT}/')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
