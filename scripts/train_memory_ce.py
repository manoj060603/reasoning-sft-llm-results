"""Distributed full SFT entry point with mathematically unchanged chunked CE."""
from reasoning.memory_ce import install
install()
from llamafactory.train.tuner import run_exp
if __name__ == '__main__':
    run_exp()
