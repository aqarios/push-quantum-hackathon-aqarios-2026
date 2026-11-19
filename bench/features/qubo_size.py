"""Feature: size class of a model by its QUBO variables, used to group the plots."""

from luna_bench.custom import BaseFeature, FeatureResult, feature
from luna_model import Model

from bench.utils import qubo_size

# Upper bound of QUBO variables -> label, in increasing order: the size classes of bench/rules.py, with mid split
# in two. Each class holds 3 to 15 of the 47 instances of bench/data/instances.yaml.
SIZE_CLASSES = [(8, "4-8"),
                (12, "9-12"),
                (16, "13-16"),
                (20, "17-20"),
                (24, "21-24")]


class QuboSizeResult(FeatureResult):
    """Size class (``value``) and number of QUBO variables of a model."""

    value: str
    qubo_variables: int


@feature
class QuboSize(BaseFeature[QuboSizeResult]):
    """Size class of a model: its QUBO variables (slack included), i.e. the qubits QAOA and VQE need."""

    def run(self, model: Model) -> QuboSizeResult:
        """Count the QUBO variables and look up the size class."""
        size = qubo_size(model)
        label = next((label for bound, label in SIZE_CLASSES if size <= bound), f">{SIZE_CLASSES[-1][0]}")
        return QuboSizeResult(value=label, qubo_variables=size)
