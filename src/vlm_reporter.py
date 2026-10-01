import logging
from typing import Optional

from src.vlm import VLMAnalyzer, VLMWorker, template_analysis

log = logging.getLogger("TrafficSystem")


class VLMReporter:
    """
    Anomaly reporter for traffic control rooms / police dashboards.

    When a local agent reports a lane that does not move despite green, the
    reporter builds the sensor context and asks the vision-language model
    (src/vlm.py) to verify it on the camera frame and write the report.
    Requests run on a background thread; without API credentials (or with
    `analyzer=None`) the rule-based template report is used instead.
    """

    def __init__(self, analyzer: Optional[VLMAnalyzer] = None):
        self.reports_generated = 0
        self.analyzer = analyzer
        self.worker = VLMWorker(analyzer) if analyzer is not None and analyzer.available else None
        self._pending = []  # contexts waiting while the worker is busy

    @property
    def mode(self):
        return "vlm" if self.worker is not None and self.analyzer.available else "template"

    def _context(self, ix_id, agent, lanes):
        env = agent.env
        return {
            "intersection": ix_id,
            "trigger": "lane_not_discharging_on_green",
            "anomalous_lanes": [int(l) for l in lanes],
            "queues": [round(float(q), 1) for q in agent.queues],
            "current_phase": int(agent.current_phase),
            "green_lanes": [int(l) for l in env.green_map[agent.current_phase]],
            "green_seconds_without_discharge": {
                int(l): int(agent.locked_queues_duration[l]) for l in lanes},
        }

    def check_and_report(self, supervisor, frames: Optional[dict] = None) -> list[dict]:
        """
        Scan all intersections managed by the supervisor and report anomalies.

        frames: optional {intersection_id: BGR camera frame} for visual checks.

        Returns a list of dicts (reports finished since the last call):
            {"intersection": id, "lanes": [...], "text": Arabic report,
             "summary": short English line (OpenCV cannot render Arabic),
             "source": "vlm" | "template", "analysis": SceneAnalysis dict}
        """
        frames = frames or {}
        for ix_id, agent in supervisor.intersections.items():
            anomalous_lanes = agent.get_anomalies()
            if len(anomalous_lanes) > 0:
                self._pending.append((frames.get(ix_id), self._context(ix_id, agent, anomalous_lanes)))

                # Prevent spam: reset lock duration after reporting once
                for lane in anomalous_lanes:
                    agent.locked_queues_duration[lane] = -50

        # Finished VLM analyses (the worker falls back to the template itself
        # when a request fails, so nothing submitted is ever lost)
        results = self.worker.poll() if self.worker is not None else []
        if self.worker is None or not self.analyzer.available:
            # Offline: template reports, immediately
            results += [template_analysis(ctx) for _, ctx in self._pending]
            self._pending = []
        elif self._pending and not self.worker.busy:
            frame, ctx = self._pending.pop(0)
            self.worker.submit(frame, ctx)

        reports = []
        for a in results:
            ctx = a.context
            reports.append({
                "intersection": ctx.get("intersection"),
                "lanes": ctx.get("anomalous_lanes", []),
                "text": a.report_ar,
                "summary": a.summary_en,
                "source": a.source,
                "analysis": a.to_dict(),
            })
            log.warning(f"[{a.source.upper()} report] {a.summary_en}\n{a.report_ar}")
            self.reports_generated += 1
        return reports

    def close(self):
        if self.worker is not None:
            self.worker.close()
