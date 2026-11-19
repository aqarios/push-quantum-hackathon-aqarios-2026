"""Feature: the use case of a model, used to group the plots."""

from luna_bench.custom import BaseFeature, FeatureResult, feature
from luna_model import Model


class UseCaseResult(FeatureResult):
    """Name of the use case."""

    value: str


@feature
class UseCase(BaseFeature[UseCaseResult]):
    """Use case of a model, taken from its name ``<use case>_<instance>`` (see bench/data/instances.yaml)."""

    def run(self, model: Model) -> UseCaseResult:
        """Return the part of the model name before the first underscore."""
        return UseCaseResult(value=model.name.split("_", 1)[0])
