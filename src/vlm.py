"""
Vision-Language Scene Analysis (Claude)
========================================
Second-level perception for rare events that the vehicle detector cannot
classify:

  * emergency vehicles (ambulance / fire / police with active lights):
    which approach they are on and whether they will go straight or left,
    so the signal controller can pre-empt the right phase;
  * incidents (collision, broken-down vehicle, blocked lane, signal fault):
    type, approach and severity, with a report for the traffic operations
    room in Arabic and a one-line English summary.

The analyzer sends the camera frame (JPEG) plus structured sensor context
(phase, per-lane queues, why the check was triggered) to Claude and gets
back JSON that follows `SCENE_SCHEMA` (structured outputs). Without an
image it writes the report from the sensor context only.

If the `anthropic` package or API credentials are missing, or a request
fails, callers fall back to `template_analysis()` so the system keeps
working offline.

Configuration: the `vlm` section of configs/config.yaml.
"""

import base64
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

log = logging.getLogger("TrafficSystem")

APPROACHES = ["north", "south", "east", "west"]
LANE_NAMES_EN = [
    "N straight", "N left", "S straight", "S left",
    "E straight", "E left", "W straight", "W left",
]
LANE_NAMES_AR = [
    "الشمالي المستقيم", "الشمالي يسار", "الجنوبي المستقيم", "الجنوبي يسار",
    "الشرقي المستقيم", "الشرقي يسار", "الغربي المستقيم", "الغربي يسار",
]

_APPROACH_ENUM = APPROACHES + ["unknown", "none"]

SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "emergency_vehicle": {
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "approach": {"type": "string", "enum": _APPROACH_ENUM},
                "movement": {"type": "string", "enum": ["straight", "left", "unknown"]},
                "evidence": {"type": "string"},
            },
            "required": ["present", "approach", "movement", "evidence"],
            "additionalProperties": False,
        },
        "incident": {
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "type": {"type": "string", "enum": [
                    "none", "collision", "breakdown", "lane_blocked",
                    "signal_malfunction", "pedestrian_hazard", "debris", "other"]},
                "approach": {"type": "string", "enum": _APPROACH_ENUM},
                "severity": {"type": "string", "enum": ["none", "low", "medium", "high"]},
            },
            "required": ["present", "type", "approach", "severity"],
            "additionalProperties": False,
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "summary_en": {"type": "string"},
        "report_ar": {"type": "string"},
    },
    "required": ["emergency_vehicle", "incident", "confidence", "summary_en", "report_ar"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You analyze a fixed traffic camera at a signalized 4-way intersection for a smart traffic-signal controller and the traffic police operations room.

Each approach has a straight lane and a left-turn lane (8 lanes). You receive a camera frame (when available) and sensor data from the vehicle detector and signal controller.

Your two jobs:
1. Emergency vehicles: report present=true only when an ambulance, fire truck or police vehicle is clearly visible with emergency markings or active lights. Give its approach and, from its lane position or indicator, whether it will go straight or turn left. The controller will give it green, so a false alarm disrupts traffic: when you are not sure, report present=false.
2. Incidents: a collision, a broken-down or stopped vehicle blocking a lane, debris, a pedestrian in the roadway, or signs of a signal malfunction. Queues that are simply waiting at a red light are NOT incidents.

Use the sensor data as context, but do not report anything the image does not support. If no image is provided, base the report only on the sensor data and say so.

summary_en: one line, at most 20 words, for the dashboard.
report_ar: 2-4 sentences in Modern Standard Arabic for the operations room: location, what was observed, and the recommended action (for example dispatching a patrol or tow truck). If nothing needs attention, say that briefly."""


# =====================================================================
#  Result type
# =====================================================================

@dataclass
class SceneAnalysis:
    emergency_present: bool = False
    emergency_approach: str = "none"
    emergency_movement: str = "unknown"
    emergency_evidence: str = ""
    incident_present: bool = False
    incident_type: str = "none"
    incident_approach: str = "none"
    severity: str = "none"
    confidence: str = "low"
    summary_en: str = ""
    report_ar: str = ""
    source: str = "template"          # "vlm" or "template"
    model: str = ""
    latency_s: float = 0.0
    context: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict, **extra):
        ev, inc = data["emergency_vehicle"], data["incident"]
        return cls(
            emergency_present=bool(ev["present"]),
            emergency_approach=ev["approach"],
            emergency_movement=ev["movement"],
            emergency_evidence=ev.get("evidence", ""),
            incident_present=bool(inc["present"]),
            incident_type=inc["type"],
            incident_approach=inc["approach"],
            severity=inc["severity"],
            confidence=data["confidence"],
            summary_en=data["summary_en"],
            report_ar=data["report_ar"],
            **extra,
        )

    def emergency_lane(self) -> Optional[int]:
        """Lane index (0-7) of a confirmed emergency vehicle, else None."""
        if not self.emergency_present or self.confidence == "low":
            return None
        if self.emergency_approach not in APPROACHES:
            return None
        base = 2 * APPROACHES.index(self.emergency_approach)
        return base + (1 if self.emergency_movement == "left" else 0)

    def to_dict(self):
        return asdict(self)


# =====================================================================
#  Offline fallback
# =====================================================================

def template_analysis(context: dict) -> SceneAnalysis:
    """Rule-based report used when the VLM is unavailable."""
    ix = context.get("intersection", "?")
    lanes = context.get("anomalous_lanes", [])
    queues = context.get("queues", [])
    if not lanes:
        return SceneAnalysis(summary_en=f"[{ix}] No anomaly.", report_ar="لا توجد ملاحظات.",
                             context=context)

    lanes_ar = " و ".join(LANE_NAMES_AR[l] for l in lanes)
    lanes_en = ", ".join(LANE_NAMES_EN[l] for l in lanes)
    max_queue = int(max((queues[l] for l in lanes), default=0)) if queues else 0
    approach = APPROACHES[lanes[0] // 2]
    report = (
        f"⚠️ **تنبيه مروري عاجل** ⚠️\n"
        f"📍 **الموقع:** {ix}\n"
        f"🛑 **المشكلة:** توقف تام للحركة في المسار/المسارات ({lanes_ar}) رغم الإشارة الخضراء.\n"
        f"📊 **الحالة:** حوالي {max_queue} مركبات لم تتحرك خلال عدة ثوانٍ من الأخضر.\n"
        f"🚓 **التوجيه المقترح:** يرجى توجيه دورية مرور للتحقق من وجود حادث أو مركبة متعطلة.\n"
    )
    return SceneAnalysis(
        incident_present=True, incident_type="lane_blocked", incident_approach=approach,
        severity="medium", confidence="medium",
        summary_en=f"[{ix}] BLOCKED despite green: {lanes_en} (~{max_queue} vehicles). Dispatch patrol.",
        report_ar=report, source="template", context=context,
    )


# =====================================================================
#  Claude-backed analyzer
# =====================================================================

class VLMUnavailable(RuntimeError):
    """No SDK / credentials, or the VLM is disabled in the config."""


class VLMAnalyzer:
    """Calls Claude with a camera frame + sensor context; returns SceneAnalysis."""

    def __init__(self, cfg: Optional[dict] = None, client=None):
        vc = (cfg or {}).get("vlm", {})
        self.enabled = bool(vc.get("enabled", True))
        self.model = vc.get("model", "claude-opus-5-5")
        self.effort = vc.get("effort", "low")
        self.max_tokens = int(vc.get("max_tokens", 8000))
        self.timeout_s = float(vc.get("timeout_s", 60))
        self.image_max_width = int(vc.get("image_max_width", 960))
        self.jpeg_quality = int(vc.get("jpeg_quality", 80))
        self.use_fallbacks = bool(vc.get("server_side_fallbacks", True))
        self._client = client
        self._disabled_reason = None if self.enabled else "disabled in config"
        if self.enabled and client is None:
            try:
                import anthropic  # noqa: F401
            except ImportError:
                self._disabled_reason = "anthropic package not installed"

    @property
    def available(self) -> bool:
        return self._disabled_reason is None

    @property
    def disabled_reason(self):
        return self._disabled_reason

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(timeout=self.timeout_s, max_retries=1)
        return self._client

    def _encode_frame(self, frame_bgr) -> str:
        import cv2
        h, w = frame_bgr.shape[:2]
        if w > self.image_max_width:
            scale = self.image_max_width / w
            frame_bgr = cv2.resize(frame_bgr, (self.image_max_width, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise ValueError("could not JPEG-encode frame")
        return base64.standard_b64encode(buf.tobytes()).decode("ascii")

    def _build_messages(self, frame_bgr, context):
        content = []
        if frame_bgr is not None:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": self._encode_frame(frame_bgr)},
            })
        text = "Sensor data (JSON):\n" + json.dumps(context, ensure_ascii=False, sort_keys=True)
        if frame_bgr is None:
            text += "\n\nNo camera image is available for this check."
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def analyze(self, frame_bgr=None, context: Optional[dict] = None) -> SceneAnalysis:
        """
        Analyze one frame. Raises VLMUnavailable when the VLM cannot be used
        at all, and anthropic API errors for transient failures.
        """
        if not self.available:
            raise VLMUnavailable(self._disabled_reason)
        context = context or {}
        import anthropic

        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=self._build_messages(frame_bgr, context),
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": SCENE_SCHEMA},
            },
        )
        if self.use_fallbacks:
            # Re-run a request that a safety classifier declines on the
            # recommended fallback model instead of returning a refusal
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")

        t0 = time.perf_counter()
        try:
            response = self._get_client().beta.messages.create(**kwargs)
        except TypeError as e:
            # Raised by the SDK before sending when no credentials are configured
            if "authentication" in str(e).lower():
                self._disabled_reason = "no Anthropic API credentials"
                raise VLMUnavailable(self._disabled_reason) from e
            raise
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                getattr(anthropic, "CredentialsError", anthropic.AuthenticationError)) as e:
            self._disabled_reason = f"credentials missing or rejected ({type(e).__name__})"
            raise VLMUnavailable(self._disabled_reason) from e
        latency = time.perf_counter() - t0

        if response.stop_reason == "refusal":
            raise RuntimeError("VLM request was declined (stop_reason=refusal)")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("VLM response was cut off (max_tokens); increase vlm.max_tokens")

        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise RuntimeError("VLM response had no text block")
        return SceneAnalysis.from_json(json.loads(text), source="vlm", model=response.model,
                                       latency_s=round(latency, 2), context=context)

    def analyze_or_fallback(self, frame_bgr=None, context: Optional[dict] = None) -> SceneAnalysis:
        """analyze(), but never raises: returns the template result on any failure."""
        context = context or {}
        if self.available:
            try:
                return self.analyze(frame_bgr, context)
            except VLMUnavailable as e:
                log.warning(f"VLM unavailable ({e}); using template reports.")
            except Exception as e:  # network, rate limit, refusal, bad JSON
                log.warning(f"VLM request failed ({type(e).__name__}: {e}); using template.")
        return template_analysis(context)


# =====================================================================
#  Non-blocking worker for real-time loops
# =====================================================================

class VLMWorker:
    """
    Runs analyses on a background thread so the video / simulation loop
    never waits for the network. At most one request is in flight; new
    submissions while busy are dropped (the next scan will catch up).
    """

    def __init__(self, analyzer: VLMAnalyzer, fallback_to_template=True):
        self.analyzer = analyzer
        self.fallback = fallback_to_template
        self._jobs: "queue.Queue" = queue.Queue(maxsize=1)
        self._results: "queue.Queue" = queue.Queue()
        self._busy = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vlm-worker", daemon=True)
        self._thread.start()

    @property
    def busy(self):
        return self._busy.is_set()

    def submit(self, frame_bgr=None, context: Optional[dict] = None) -> bool:
        if self._busy.is_set():
            return False
        frame = None if frame_bgr is None else np.array(frame_bgr, copy=True)
        try:
            self._busy.set()
            self._jobs.put_nowait((frame, context or {}))
            return True
        except queue.Full:
            return False

    def poll(self):
        """Return all analyses finished since the last call."""
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def _run(self):
        while not self._stop.is_set():
            try:
                frame, context = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if self.fallback:
                    result = self.analyzer.analyze_or_fallback(frame, context)
                else:
                    result = self.analyzer.analyze(frame, context)
                self._results.put(result)
            except Exception as e:
                log.warning(f"VLM worker error: {type(e).__name__}: {e}")
            finally:
                self._busy.clear()

    def close(self, timeout=2.0):
        self._stop.set()
        self._thread.join(timeout=timeout)
