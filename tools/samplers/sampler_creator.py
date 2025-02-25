from typing import Optional

from samplers import Sampler, RandomSampler, GreedySampler, YangSampler

class SamplerCreator:
    """
    Initialize a sampler.
    """

    def __init__(self,
                 sampler: str,
                 score_field: Optional[str] = None) -> None:
        self.sampler = sampler
        self.score_field = score_field

    def get_sampler(self) -> Sampler:

        samplers = {
            "greedy": GreedySampler,
            "random": RandomSampler,
            "yang": YangSampler
        }

        return samplers[self.sampler](score_field = self.score_field)