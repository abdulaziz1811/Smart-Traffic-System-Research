"""
Tests for the vision-language scene analyzer (no network: fake client).

Run with:  python -m pytest tests/
"""

import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vlm import (SCENE_SCHEMA, SceneAnalysis, VLMAnalyzer, VLMUnavailable, VLMWorker,
                     template_analysis)

AMBULANCE = {
    "emergency_vehicle": {"present": True, "approach": "east", "movement": "left",
                          "evidence": "white van with red cross and flashing lights"},
    "incident": {"present": False, "type": "none", "approach": "none", "severity": "none"},
    "confidence": "high",
    "summary_en": "Ambulance approaching from the east, turning left.",
    "report_ar": "سيارة إسعاف قادمة من الاتجاه الشرقي وتنوي الانعطاف يساراً.",
}


class FakeClient:
    """Mimics client.beta.messages.create and records the request."""

    def __init__(self, payload=AMBULANCE, stop_reason="end_turn", delay=0.0):
        self.calls = []
        self.payload, self.stop_reason, self.delay = payload, stop_reason, delay
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        time.sleep(self.delay)
        return SimpleNamespace(
            stop_reason=self.stop_reason, model=kwargs["model"],
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=json.dumps(self.payload, ensure_ascii=False))])


def frame():
    return np.zeros((540, 960, 3), dtype=np.uint8)


def test_request_uses_image_structured_output_and_fallbacks():
    client = FakeClient()
    a = VLMAnalyzer({"vlm": {"model": "claude-opus-5-5", "effort": "low"}}, client=client)
    res = a.analyze(frame(), {"intersection": "X"})
    req = client.calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"]["format"]["schema"] == SCENE_SCHEMA
    assert req["output_config"]["effort"] == "low"
    assert req["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in req["betas"]
    kinds = [b["type"] for b in req["messages"][0]["content"]]
    assert kinds == ["image", "text"]
    assert res.source == "vlm" and res.emergency_present


def test_text_only_request_without_frame():
    client = FakeClient()
    VLMAnalyzer({}, client=client).analyze(None, {"intersection": "X"})
    kinds = [b["type"] for b in client.calls[0]["messages"][0]["content"]]
    assert kinds == ["text"]


def test_emergency_lane_mapping():
    a = SceneAnalysis.from_json(AMBULANCE)
    assert a.emergency_lane() == 5            # east left
    low = SceneAnalysis.from_json({**AMBULANCE, "confidence": "low"})
    assert low.emergency_lane() is None        # never pre-empt on a low-confidence call


def test_refusal_falls_back_to_template():
    a = VLMAnalyzer({}, client=FakeClient(stop_reason="refusal"))
    with pytest.raises(RuntimeError):
        a.analyze(frame(), {})
    res = a.analyze_or_fallback(frame(), {"intersection": "X", "anomalous_lanes": [4], "queues": [0] * 8})
    assert res.source == "template" and res.incident_present


def test_missing_credentials_disables_vlm(monkeypatch):
    pytest.importorskip("anthropic")
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", "/nonexistent")  # no `ant auth login` profile
    a = VLMAnalyzer({})
    try:
        a.analyze(None, {})
    except VLMUnavailable:
        assert not a.available
        res = a.analyze_or_fallback(None, {"intersection": "X", "anomalous_lanes": [0], "queues": [5] * 8})
        assert res.source == "template"
    except Exception as e:  # credentials exist in this environment: nothing to check
        pytest.skip(f"credentials available ({type(e).__name__})")


def test_disabled_in_config():
    a = VLMAnalyzer({"vlm": {"enabled": False}}, client=FakeClient())
    assert not a.available
    with pytest.raises(VLMUnavailable):
        a.analyze(None, {})


def test_worker_is_non_blocking_and_drops_while_busy():
    w = VLMWorker(VLMAnalyzer({}, client=FakeClient(delay=0.3)))
    t0 = time.perf_counter()
    assert w.submit(frame(), {"intersection": "X"})
    assert time.perf_counter() - t0 < 0.1
    assert not w.submit(frame(), {})          # busy -> dropped
    deadline = time.time() + 3
    results = []
    while not results and time.time() < deadline:
        results = w.poll()
        time.sleep(0.05)
    w.close()
    assert len(results) == 1 and results[0].emergency_lane() == 5


def test_template_analysis():
    res = template_analysis({"intersection": "Main_1", "anomalous_lanes": [4], "queues": [0, 0, 0, 0, 12, 0, 0, 0]})
    assert res.incident_present and res.incident_approach == "east"
    assert "E straight" in res.summary_en and "12" in res.summary_en


def test_reporter_keeps_report_when_vlm_turns_out_unavailable(monkeypatch):
    """First request fails (no credentials) -> its template result must still be reported."""
    from src.config import load_config
    from src.environment import TrafficSignalEnv
    from src.intersection import LocalIntersectionAgent
    from src.supervisor import CentralSupervisor
    from src.vlm_reporter import VLMReporter

    class NoCredentialsClient:
        beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(
            TypeError("Could not resolve authentication method."))))

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "configs", "config.yaml"))
    env = TrafficSignalEnv(cfg)
    env.reset(seed=0)
    agent = LocalIntersectionAgent("Main_Intersection_1", env, None)
    sup = CentralSupervisor()
    sup.register_intersection(agent)
    rep = VLMReporter(VLMAnalyzer(cfg, client=NoCredentialsClient()))

    agent.locked_queues_duration[4] = 99          # anomaly on lane 4
    env.queues[4] = 12
    reports = rep.check_and_report(sup)           # submitted to the worker
    deadline = time.time() + 3
    while not reports and time.time() < deadline:
        time.sleep(0.05)
        reports = rep.check_and_report(sup)
    rep.close()
    assert reports and reports[0]["source"] == "template" and reports[0]["lanes"] == [4]
    assert rep.mode == "template"
