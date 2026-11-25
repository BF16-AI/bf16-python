import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import bf16ai


class _MockHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            self._send_json({"data": [{"id": bf16ai.MODEL_GLM_53}]})
        elif self.path == "/v1/me":
            if self.headers.get("Authorization") != "Bearer sk-bf16-test":
                self.send_response(401)
                self.end_headers()
                return
            self._send_json({"balance": {"available_usd": 10.5}, "tier": {"id": 1, "name": "Tier 1"}})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._send_json({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    def _send_json(self, obj):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class TestBf16Client(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _MockHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_models_me_chat_and_stream(self):
        client = bf16ai.Client(api_key="sk-bf16-test", base_url=f"http://127.0.0.1:{self.port}/v1")
        models = client.list_models()
        self.assertEqual(models[0]["id"], bf16ai.MODEL_GLM_53)

        me = client.me()
        self.assertEqual(me["balance"]["available_usd"], 10.5)

        res = client.chat([{"role": "user", "content": "ping"}])
        self.assertEqual(res["choices"][0]["message"]["content"], "ok")

        chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "hi")

    def test_missing_key_raises(self):
        client = bf16ai.Client(api_key="", base_url=f"http://127.0.0.1:{self.port}/v1")
        with self.assertRaises(bf16ai.Bf16Error):
            client.me()


if __name__ == "__main__":
    unittest.main()
