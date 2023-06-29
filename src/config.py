from enum import Enum

class QuantizedType(Enum):
    VECTOR = 1
    PRODUCT = 2

class Config:
    RESOURCE_DIR = 'resources'

    SEED_NUM = 10190350440
    SEED_MOD = (1 << 32)
    SEED = SEED_NUM % SEED_MOD

    SPLIT = "=" * 30