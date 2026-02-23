import logging

log = logging.getLogger("TrafficSystem")


class CentralSupervisor:
    """
    Central Supervisor representing the higher-level agent managing multiple local intersections
    in the neighborhood. Responsible for the 'Green Wave' and routing anomalies.
    """

    def __init__(self):
        self.intersections = {}  # Map ID -> LocalIntersectionAgent
        self.green_wave_active = False
        self.ambulance_path_plan = (
            []
        )  # List of intersection IDs the ambulance will cross

    def register_intersection(self, intersection_agent):
        self.intersections[intersection_agent.id] = intersection_agent
        log.info(f"Supervisor registered intersection: {intersection_agent.id}")

    def handle_ambulance_intention(self, source_intersection_id, intention):
        """
        Receives the predicted intention of an ambulance from a local agent
        and propagates the Green Wave to the next expected intersections.
        """
        log.warning(
            f"[Supervisor] Received emergency signal from {source_intersection_id}. Intention: {intention}"
        )
        self.green_wave_active = True

        # Simple logical mapping defined by neighborhood layout
        # Example: 'intersection_1' going 'straight' goes to 'intersection_2'
        next_intersection = self._predict_next_route(source_intersection_id, intention)

        if next_intersection and next_intersection in self.intersections:
            target_agent = self.intersections[next_intersection]
            log.warning(
                f"[Supervisor] Deploying GREEN WAVE to next intersection: {next_intersection}"
            )

            # Pre-emptively trigger emergency mode on the next intersection
            # Assuming it enters on the opposite lane (simplified logical mapping)
            expected_entry_lane = self._get_expected_entry_lane(intention)
            target_agent.detect_ambulance(lane=expected_entry_lane)
            self.ambulance_path_plan.append((source_intersection_id, next_intersection))

    def handle_route_correction(self, source_intersection_id, actual_intention):
        """
        Self-correcting mechanism flexibly updating the Green Wave if the ambulance
        driver abruptly changes lane/indicator.
        """
        log.info(
            f"[Supervisor] Route correction initiated from {source_intersection_id}. New intention: {actual_intention}"
        )
        # 1. Clear previous green wave
        for past_source, past_target in self.ambulance_path_plan:
            if past_source == source_intersection_id:
                if past_target in self.intersections:
                    self.intersections[past_target].clear_ambulance()
                    log.info(f"[Supervisor] Cancelled Green Wave at {past_target}")

        self.ambulance_path_plan = []

        # 2. Deploy new green wave
        self.handle_ambulance_intention(source_intersection_id, actual_intention)

    def _predict_next_route(self, current_id, intention):
        """Predict the next intersection in the grid. Hardcoded 1D street for demo."""
        if current_id == "Main_Intersection_1" and intention == "straight":
            return "Main_Intersection_2"
        elif current_id == "Main_Intersection_1" and intention == "left":
            return "Side_Street_Intersection"
        return None

    def _get_expected_entry_lane(self, intention):
        """Get the expected lane index the ambulance will arrive at in the next intersection."""
        if intention == "straight":
            return 0  # N Straight
        return 4  # E Straight
