import cv2
import numpy as np
from stable_baselines3 import PPO

from src.config import bootstrap
from src.environment import TrafficSignalEnv
from src.intersection import LocalIntersectionAgent
from src.supervisor import CentralSupervisor
from src.vlm_reporter import VLMReporter


def get_light_color(phase, target_lanes):
    """Return color for the specific lanes given the phase."""
    green_map = {
        0: [0, 2],  # NS Straight
        1: [1, 3],  # NS Left
        2: [4, 6],  # EW Straight
        3: [5, 7],  # EW Left
    }
    for p, lanes in green_map.items():
        if p == phase:
            for lane in target_lanes:
                if lane in lanes:
                    return (0, 255, 0)  # Green
    return (0, 0, 255)  # Red


def draw_intersection(img, x_offset, y_offset, agent, title):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, title, (x_offset, y_offset), font, 0.7, (255, 200, 0), 2)

    queues = agent.queues
    phase = agent.current_phase

    # Status modes
    if agent.emergency_mode:
        mode_text = "EMERGENCY OVERRIDE!"
        mode_color = (0, 0, 255)  # Red text for emergency
    else:
        mode_text = "AI RL CONTROL"
        mode_color = (0, 255, 255)

    cv2.putText(
        img, f"Mode: {mode_text}", (x_offset, y_offset + 30), font, 0.6, mode_color, 2
    )

    # Phase Info
    cv2.putText(
        img,
        f"Current Phase: {phase}",
        (x_offset, y_offset + 60),
        font,
        0.6,
        (255, 255, 255),
        1,
    )

    # NS / EW Status
    ns_str_color = get_light_color(phase, [0, 2])
    ns_left_color = get_light_color(phase, [1, 3])
    ew_str_color = get_light_color(phase, [4, 6])
    ew_left_color = get_light_color(phase, [5, 7])

    cv2.putText(
        img,
        f"N/S Straight: {'GREEN' if ns_str_color[1]>0 else 'RED'}",
        (x_offset, y_offset + 90),
        font,
        0.5,
        ns_str_color,
        1,
    )
    cv2.putText(
        img,
        f"N/S Left   : {'GREEN' if ns_left_color[1]>0 else 'RED'}",
        (x_offset, y_offset + 115),
        font,
        0.5,
        ns_left_color,
        1,
    )

    cv2.putText(
        img,
        f"E/W Straight: {'GREEN' if ew_str_color[1]>0 else 'RED'}",
        (x_offset + 200, y_offset + 90),
        font,
        0.5,
        ew_str_color,
        1,
    )
    cv2.putText(
        img,
        f"E/W Left   : {'GREEN' if ew_left_color[1]>0 else 'RED'}",
        (x_offset + 200, y_offset + 115),
        font,
        0.5,
        ew_left_color,
        1,
    )

    # Queues
    q_ns = int(queues[0] + queues[2] + queues[1] + queues[3])
    q_ew = int(queues[4] + queues[6] + queues[5] + queues[7])
    cv2.putText(
        img,
        f"Queues - NS: {q_ns}, EW: {q_ew}",
        (x_offset, y_offset + 145),
        font,
        0.5,
        (200, 200, 200),
        1,
    )

    cv2.rectangle(
        img,
        (x_offset - 10, y_offset - 25),
        (x_offset + 420, y_offset + 160),
        (100, 100, 100),
        2,
    )


def main():
    cfg, log, device = bootstrap("configs/config.yaml")

    # Fallback to load model safely
    try:
        rl_model = PPO.load("models/rl_agents/final_ppo_agent")
    except Exception:
        log.warning(
            "PPO model 'final_ppo_agent' not found. Ensure you have trained the model or the path is correct. Running with random actions for demo."
        )
        rl_model = None

    # Setup the multi-agent system
    env1 = TrafficSignalEnv(cfg)
    env2 = TrafficSignalEnv(cfg)

    obs1, _ = env1.reset()
    obs2, _ = env2.reset()

    # Force heavy traffic to demonstrate VLM Anomaly Reporter faster
    env1.arrivals = np.array([0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1])

    # Wrap with our new LocalAgents
    agent1 = LocalIntersectionAgent("Main_Intersection_1", env1, rl_model)
    agent2 = LocalIntersectionAgent("Main_Intersection_2", env2, rl_model)

    # Create Supervisor and VLM
    supervisor = CentralSupervisor()
    supervisor.register_intersection(agent1)
    supervisor.register_intersection(agent2)

    vlm_reporter = VLMReporter()

    # UI Setup
    width, height = 1000, 700

    sim_step = 0
    active_reports: list[str] = []

    log.info("Starting Full Architecture Demo...")

    while True:
        sim_step += 1
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Scenario Timeline:
        # Step 50: Ambulance arrives at Intersection 1 going straight
        if sim_step == 50:
            agent1.detect_ambulance(
                lane=0, is_blinking_left_indicator=False
            )  # N Straight
            supervisor.handle_ambulance_intention(agent1.id, agent1.ambulance_intention)

        # Step 150: Ambulance passes Int 1
        if sim_step == 150:
            agent1.clear_ambulance()

        # Step 250: Ambulance passes Int 2 (which was pre-emptively on Green Wave)
        if sim_step == 250:
            agent2.clear_ambulance()

        # 1. RL Agent decisions
        if rl_model:
            action1 = agent1.get_action(obs1)
            action2 = agent2.get_action(obs2)
        else:
            action1 = (
                env1.action_space.sample()  # type: ignore
                if not agent1.emergency_mode
                else agent1.get_action(obs1)
            )
            action2 = (
                env2.action_space.sample()  # type: ignore
                if not agent2.emergency_mode
                else agent2.get_action(obs2)
            )

        # 2. Step Environments
        obs1, _, _, _, _ = agent1.step(action1)
        obs2, _, _, _, _ = agent2.step(action2)

        # 3. Check VLM Anomalies
        # We only check every 10 steps to simulate periodic scanning
        if sim_step % 10 == 0:
            new_reports = vlm_reporter.check_and_report(supervisor)
            if new_reports:
                active_reports.extend(new_reports)
                # Keep only last 2 reports to avoid clutter
                while len(active_reports) > 2:
                    active_reports.pop(0)

        # --- Drawing UI ---
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Header
        cv2.putText(
            frame,
            "SMART TRAFFIC SYSTEM architecture v2.0",
            (30, 40),
            font,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Local Agents + Central Supervisor + VLM Reporter",
            (30, 70),
            font,
            0.6,
            (150, 150, 150),
            1,
        )

        # Intersections
        draw_intersection(frame, 30, 150, agent1, "INTERSECTION 1 (Local Agent)")
        draw_intersection(frame, 500, 150, agent2, "INTERSECTION 2 (Local Agent)")

        # Supervisor Status Panel
        cv2.rectangle(frame, (30, 350), (920, 450), (40, 40, 40), -1)
        cv2.putText(
            frame, "CENTRAL SUPERVISOR STATUS", (40, 380), font, 0.7, (0, 255, 255), 2
        )

        wave_status = (
            "ACTIVE (Connecting Route)"
            if supervisor.green_wave_active
            else "Standby (Normal Operations)"
        )
        wave_color = (0, 255, 0) if supervisor.green_wave_active else (200, 200, 200)
        cv2.putText(
            frame, f"Green Wave: {wave_status}", (40, 415), font, 0.6, wave_color, 1
        )

        if supervisor.ambulance_path_plan:
            path_str = " -> ".join(
                [f"{s} to {t}" for s, t in supervisor.ambulance_path_plan]
            )
            cv2.putText(
                frame,
                f"Active Emergency Plan: {path_str}",
                (40, 440),
                font,
                0.5,
                (0, 150, 255),
                1,
            )

        # VLM Reports Panel
        cv2.rectangle(frame, (30, 470), (920, 670), (20, 20, 60), -1)
        cv2.putText(
            frame,
            "VLM ANOMALY REPORTER (Live Ops Room Stream)",
            (40, 500),
            font,
            0.7,
            (255, 100, 100),
            2,
        )

        y_rep = 530
        for i, rep in enumerate(active_reports):
            # Split the multi-line string report
            lines = rep.split("\n")
            for line in lines:
                if line.strip():
                    cv2.putText(
                        frame,
                        line.replace("**", ""),
                        (40, y_rep),
                        font,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    y_rep += 20
            y_rep += 10  # space between reports

        cv2.imshow("Full System Architecture Demo", frame)

        # Frame delay
        key = cv2.waitKey(50)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
