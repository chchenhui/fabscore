# Gradient-evaluation budget counter for compute-fair comparisons.
# Tracks total forward/backward passes across optimizers to ensure
# equal compute budgets between Adam and L-BFGS variants.
# One "evaluation" = one (forward + backward) pass on a collocation/data set.


class BudgetTracker:

    def __init__(self, budget):
        self.budget = budget
        self._count = 0

    @property
    def count(self):
        return self._count

    def increment(self, n=1):
        self._count += n

    def remaining(self):
        return max(0, self.budget - self._count)

    def exhausted(self):
        return self._count >= self.budget
