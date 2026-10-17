import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shear_agent.agent import (  # noqa: E402
    LLMClient,
    LLMConfigError,
    LLMError,
    MockBackend,
    MockLLM,
    dispatch,
    run_agent,
)
from shear_agent.agent.llm import (  # noqa: E402
    HTTPStatusError,
    NetworkError,
    from_openai_reply,
    load_dotenv,
    to_openai_messages,
)

SECRET = "fake-test-key-0123456789-SECRET"   # deliberately not shaped like a real key
ANTHROPIC_OK = {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 2}}


class FakeTransport:
    """Replays a list of outcomes (dict response or exception) and records requests."""

    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def __call__(self, url, headers, payload, timeout):
        self.requests.append((url, headers, payload))
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class TestClientSafety(unittest.TestCase):
    def test_missing_key_is_a_config_error(self):
        with self.assertRaises(LLMConfigError):
            LLMClient(provider="anthropic", api_key=None)
        with self.assertRaises(LLMConfigError):
            LLMClient(provider="openai", api_key="x")          # no model assumed for openai
        with self.assertRaises(LLMConfigError):
            LLMClient(provider="nope", api_key="x")

    def test_key_never_in_repr_or_errors(self):
        body = json.dumps({"error": {"type": "authentication_error", "message": f"bad key {SECRET}"}})
        client = LLMClient(api_key=SECRET, transport=FakeTransport([HTTPStatusError(401, body)]),
                           sleep=lambda s: None)
        self.assertNotIn(SECRET, repr(client))
        with self.assertRaises(LLMError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn(SECRET, str(ctx.exception))

    def test_retries_overload_then_succeeds(self):
        sleeps = []
        t = FakeTransport([HTTPStatusError(529), NetworkError("reset"), ANTHROPIC_OK])
        client = LLMClient(api_key=SECRET, transport=t, sleep=sleeps.append)
        reply = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(reply.text, "hi")
        self.assertEqual(len(sleeps), 2)
        self.assertEqual(reply.usage, {"input_tokens": 5, "output_tokens": 2})
        url, headers, payload = t.requests[0]
        self.assertEqual(headers["x-api-key"], SECRET)
        self.assertEqual(payload["max_tokens"], 1024)

    def test_retry_after_is_honoured_and_retries_are_capped(self):
        sleeps = []
        t = FakeTransport([HTTPStatusError(429, retry_after="3")] * 3)
        client = LLMClient(api_key=SECRET, transport=t, sleep=sleeps.append, max_retries=2)
        with self.assertRaises(LLMError):
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(sleeps, [3.0, 3.0])

    def test_dotenv_does_not_override_environment(self):
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / ".env"
            env.write_text('# comment\nSHEAR_TEST_A="quoted"\nSHEAR_TEST_B=fromfile\nSHEAR_TEST_C=\n')
            with mock.patch.dict(os.environ, {"SHEAR_TEST_B": "fromenv"}, clear=False):
                load_dotenv(env)
                self.assertEqual(os.environ["SHEAR_TEST_A"], "quoted")
                self.assertEqual(os.environ["SHEAR_TEST_B"], "fromenv")
                self.assertNotIn("SHEAR_TEST_C", os.environ)
            os.environ.pop("SHEAR_TEST_A", None)


class TestOpenAICompat(unittest.TestCase):
    def test_history_conversion(self):
        history = [
            {"role": "user", "content": "pull book 7"},
            {"role": "assistant", "content": [{"type": "text", "text": "checking"},
                                              {"type": "tool_use", "id": "t1", "name": "shear_margin",
                                               "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                          "content": "{\"Q\": 0.3}"}]},
        ]
        msgs = to_openai_messages(history, system="sys")
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "tool"])
        self.assertEqual(msgs[2]["tool_calls"][0]["function"]["name"], "shear_margin")
        self.assertEqual(msgs[3]["tool_call_id"], "t1")

    def test_reply_parsing_and_client(self):
        data = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "set_margin", "arguments": "{\"q_star\": 0.3}"}}]}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        reply = from_openai_reply(data)
        self.assertEqual(reply.tool_calls[0]["input"], {"q_star": 0.3})
        self.assertEqual(reply.stop_reason, "tool_use")
        t = FakeTransport([data])
        client = LLMClient(provider="openai", model="some-model", api_key=SECRET, transport=t,
                           base_url="http://localhost:8000/v1/")
        client.chat([{"role": "user", "content": "x"}], tools=[{"name": "a", "input_schema": {}}])
        url, headers, payload = t.requests[0]
        self.assertEqual(url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(headers["authorization"], f"Bearer {SECRET}")
        self.assertEqual(payload["tools"][0]["type"], "function")


class TestAgentLoop(unittest.TestCase):
    def test_mock_episode_recovers_from_slip(self):
        result = run_agent("pull book 7", MockBackend(), MockLLM())
        tools = [s["tool"] for s in result.steps]
        self.assertEqual(result.stopped, "done")
        self.assertIn("set_margin", tools)
        self.assertEqual(result.steps[-1]["output"]["Q"], 0.3)
        self.assertIn("0.3", result.final_text)

    def test_turn_limit(self):
        result = run_agent("pull book 7", MockBackend(), MockLLM(), max_turns=2)
        self.assertEqual(result.stopped, "max_turns")
        self.assertEqual(len(result.steps), 2)

    def test_tool_inputs_are_validated(self):
        be = MockBackend()
        self.assertFalse(dispatch(be, "execute_skill", {"skill": "fly"})["ok"])
        self.assertFalse(dispatch(be, "set_margin", {"q_star": 2.0})["ok"])
        self.assertFalse(dispatch(be, "set_margin", {"q_star": "0.3"})["ok"])
        self.assertFalse(dispatch(be, "set_margin", {})["ok"])
        self.assertFalse(dispatch(be, "contact_state", {"force": 100})["ok"])
        self.assertTrue(dispatch(be, "set_margin", {"q_star": 0.3})["ok"])
        self.assertEqual(be.q_star, 0.3)


if __name__ == "__main__":
    unittest.main()
