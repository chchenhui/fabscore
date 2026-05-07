"""
Base interface for external measurement probes.
"""


class Probe:
    """
    Abstract measurement probe.

    A probe evaluates a proposed model state using information
    external to the training objective (e.g. validation probes).
    """

    def evaluate(self, model):
        """
        Evaluate the current model state.

        Parameters
        ----------
        model : torch.nn.Module
            Model to be evaluated.

        Returns
        -------
        float
            Scalar measurement value.
        """
        raise NotImplementedError
