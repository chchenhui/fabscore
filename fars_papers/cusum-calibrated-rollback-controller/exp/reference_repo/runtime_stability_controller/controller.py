"""
Runtime stability controller.

This module implements the runtime accept / rollback mechanism described
in Algorithm 1 of:

    "Automatic Stability and Recovery for Neural Network Training"
    Barak Or, 2026

The controller supervises optimizer-proposed updates using an external
measurement signal (probe) and enforces a bounded-degradation safety
invariant via rollback to a previously accepted safe state.
"""

from runtime_stability_controller.exceptions import StabilityViolation


class StabilityController:
    """
    Runtime stability controller (Algorithm 1).

    The controller operates as an external supervisory layer that:
    1. Observes optimizer-proposed updates
    2. Evaluates them using an external measurement probe
    3. Accepts or rejects updates based on an innovation threshold
    4. Restores the last safe state upon rejection

    Importantly, the optimizer update rule itself is never modified.
    """

    def __init__(self, probe, snapshot_manager, threshold, smoothing=None):
        """
        Parameters
        ----------
        probe : Probe
            External measurement probe y(·) used to evaluate proposed updates.
        snapshot_manager : SnapshotManager
            Manages saving and restoring safe snapshots (θ_safe, O_safe).
        threshold : float
            Acceptance threshold ε for the innovation signal.
        smoothing : float, optional
            Exponential smoothing factor α ∈ (0, 1) for the reference signal.
        """
        self.probe = probe
        self.snapshot_manager = snapshot_manager
        self.threshold = float(threshold)
        self.smoothing = smoothing

        # Reference signal \hat{y}_t in Algorithm 1
        self._reference_value = None

        # Diagnostics (useful for logging / analysis)
        self.last_measurement = None
        self.last_innovation = None
        self.last_accepted = None

    @property
    def reference_value(self):
        """Return the current reference signal value."""
        return self._reference_value

    # ------------------------------------------------------------------
    # Algorithm 1 — Initialization
    # ------------------------------------------------------------------

    def initialize(self, model, optimizer):
        """
        Initialize the controller state.

        Corresponds to lines 1–3 of Algorithm 1:
            - Compute initial probe measurement y(θ_0)
            - Initialize reference signal \hat{y}_0
            - Store initial safe snapshot (θ_safe, O_safe)

        This method must be called once before training begins.
        """
        initial_value = float(self.probe.evaluate(model))

        self._reference_value = initial_value
        self.snapshot_manager.save(model, optimizer)

        self.last_measurement = initial_value
        self.last_innovation = 0.0
        self.last_accepted = True

    # ------------------------------------------------------------------
    # Algorithm 1 — Runtime Supervision Loop
    # ------------------------------------------------------------------

    def step(self, model, optimizer):
        """
        Perform a supervised optimizer step.

        This method implements the main loop of Algorithm 1 (lines 4–17).

        Assumes gradients have already been computed via loss.backward().

        Returns
        -------
        bool
            True if the proposed update was accepted,
            False if rollback was triggered.
        """
        if self._reference_value is None:
            raise RuntimeError(
                "StabilityController not initialized. "
                "Call initialize(model, optimizer) before training."
            )

        # --------------------------------------------------------------
        # Algorithm 1, line 3:
        # Save current safe snapshot (θ_safe, O_safe)
        # --------------------------------------------------------------
        self.snapshot_manager.save(model, optimizer)

        # --------------------------------------------------------------
        # Algorithm 1, line 5–6:
        # Optimizer proposes and applies update
        #   θ_prop = θ_t + Δθ_t
        # --------------------------------------------------------------
        optimizer.step()

        # --------------------------------------------------------------
        # Algorithm 1, line 7:
        # Measure proposed state using external probe
        #   ν_t = y(θ_prop) − \hat{y}_t
        # --------------------------------------------------------------
        measurement = float(self.probe.evaluate(model))
        innovation = measurement - self._reference_value

        self.last_measurement = measurement
        self.last_innovation = innovation

        # --------------------------------------------------------------
        # Algorithm 1, line 8:
        # Acceptance test based on innovation threshold ε
        # --------------------------------------------------------------
        if innovation <= self.threshold:
            # ----------------------------------------------------------
            # Algorithm 1, lines 9–12:
            # Accept update and update reference signal
            # ----------------------------------------------------------
            self._update_reference(measurement)
            self.last_accepted = True
            return True

        # --------------------------------------------------------------
        # Algorithm 1, lines 13–15:
        # Reject update and restore last safe snapshot
        # --------------------------------------------------------------
        self.snapshot_manager.restore(model, optimizer)
        self.last_accepted = False

        return False

    # ------------------------------------------------------------------
    # Reference signal update
    # ------------------------------------------------------------------

    def _update_reference(self, value):
        """
        Update the reference signal \hat{y}_t after an accepted step.

        Implements the exponential smoothing rule described in Algorithm 1:
            \hat{y}_{t+1} = (1 − α) \hat{y}_t + α y(θ_{t+1})
        """
        value = float(value)

        if self.smoothing is None:
            self._reference_value = value
            return

        alpha = float(self.smoothing)
        if not (0.0 < alpha < 1.0):
            raise ValueError("smoothing parameter α must lie in (0, 1)")

        self._reference_value = (1.0 - alpha) * self._reference_value + alpha * value
