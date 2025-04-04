from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config import Config, map_options
from config.gpt.models import GPTConfig, gpt_options


class SAEVariant(str, Enum):
    STANDARD = "standard"
    STANDARDLR = 'standardLR'
    STANDARD_V2 = "standard_v2"
    GATED = "gated"
    GATED_V2 = "gated_v2"
    JUMP_RELU = "jumprelu"
    TOPK = "topk"



@dataclass
class SAEConfig(Config):
    gpt_config: GPTConfig = field(default_factory=GPTConfig)
    n_features: tuple = ()  # Number of features in each layer
    sae_variant: SAEVariant = SAEVariant.STANDARD
    top_k: Optional[tuple[int, ...]] = None  # Required for topk variant
    rank_bound: Optional[int] = None  # Low rank bound for standardLR variant

    @property
    def block_size(self) -> int:
        return self.gpt_config.block_size

    @staticmethod
    def dict_factory(fields: list) -> dict:
        """
        Only export n_features and sae_variant.
        """
        whitelisted_fields = ("n_features", "sae_variant", "top_k", "rank_bound")
        return {k: v for (k, v) in fields if k in whitelisted_fields and v is not None}


# SAE configuration options
sae_options: dict[str, SAEConfig] = map_options(
    SAEConfig(
        name="standardx8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.STANDARD,
    ),
    SAEConfig(
        name="standardLRx8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.STANDARDLR,
    ),
    SAEConfig(
        name="topk-10-x8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.TOPK,
        top_k=(10, 10, 10, 10, 10),
    ),
    SAEConfig(
        name="jumprelu-x8.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        n_features=tuple(64 * n for n in (8, 8, 8, 8, 8)),
        sae_variant=SAEVariant.JUMP_RELU,
    ),
    SAEConfig(
        name="jumprelu-x16.shakespeare_64x4",
        gpt_config=gpt_options["ascii_64x4"],
        # Only the penultimate layer needs the full x16 expansion factor
        n_features=tuple(64 * n for n in (4, 4, 4, 16, 4)),
        sae_variant=SAEVariant.JUMP_RELU,
    ),
    SAEConfig(
        name="jumprelu-x32.stories_256x4",
        gpt_config=gpt_options["tiktoken_256x4"],
        n_features=tuple(256 * n for n in (32, 32, 32, 32, 32)),
        sae_variant=SAEVariant.JUMP_RELU,
    ),
)
