import logging

log = logging.getLogger("TrafficSystem")


class VLMReporter:
    """
    Vision-Language Model anomaly reporter for automated communication
    with traffic control rooms/police dashboards.
    """

    def __init__(self):
        self.reports_generated = 0

    def check_and_report(self, supervisor) -> list[str]:
        """
        Scan all intersections managed by the supervisor and generate natural
        language reports if an anomaly is detected (e.g., stuck traffic or broken light).
        """
        reports = []
        for ix_id, agent in supervisor.intersections.items():
            anomalous_lanes = agent.get_anomalies()
            if len(anomalous_lanes) > 0:
                report = self.generate_anomaly_report(
                    ix_id, anomalous_lanes, agent.queues
                )
                reports.append(report)

                # Prevent spam: reset lock duration after reporting once
                for lane in anomalous_lanes:
                    agent.locked_queues_duration[lane] = -50

        return reports

    def generate_anomaly_report(self, intersection_id, anomalous_lanes, queues):
        """
        Simulates an LLM/VLM processing the visual context into a human-readable report.
        In a real scenario, this would format an API call to an LLM passing the camera feed frame.
        """
        # Direction mapping
        directions = {
            0: "الشمالي المستقيم",
            1: "الشمالي يسار",
            2: "الجنوبي المستقيم",
            3: "الجنوبي يسار",
            4: "الشرقي المستقيم",
            5: "الشرقي يسار",
            6: "الغربي المستقيم",
            7: "الغربي يسار",
        }

        lanes_text = " و ".join(
            [directions.get(lane, f"مسار {lane}") for lane in anomalous_lanes]
        )
        max_queue = int(max([queues[lane] for lane in anomalous_lanes]))

        report = (
            f"⚠️ **تنبيه أمني مروري عاجل** ⚠️\n"
            f"📍 **الموقع:** {intersection_id}\n"
            f"🛑 **المشكلة:** رصد نموذج الرؤية اختناقاً وتوقفاً تاماً للحركة في المسار/المسارات ({lanes_text}).\n"
            f"📊 **الحالة:** تم احتجاز حوالي {max_queue} مركبات دون تحرك لعدة دورات رغم الإشارة الخضراء.\n"
            f"🚓 **التوجيه المقترح:** يرجى توجيه دورية مرور فوراً للتحقق من وجود أعطال أو حوادث في الموقع.\n"
        )

        log.error(f"[VLM Reporter generated a report for {intersection_id}]")
        self.reports_generated += 1
        return report
