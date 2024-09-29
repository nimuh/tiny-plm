from dataclasses import dataclass


@dataclass
class PLMConfig:
    block_size: int = 1024
    vocab_size: int = 20
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 256
