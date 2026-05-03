from dataclasses import dataclass
from pathlib import Path

from appdirs import user_cache_dir

CACHE_DIR = Path(__file__).parents[2] / "resources"

if not CACHE_DIR.exists():
    CACHE_DIR = Path(user_cache_dir(appname="semgraph"))

LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class Config:
    # Directories
    cache_dir: Path = CACHE_DIR
    # Logging
    logging_format: str = LOGGING_FORMAT
    log_to_console: bool = True
    log_to_file: bool = True
    # ML/DL
    device: str = "auto"
    batch_size: int = 1


config = Config()


if __name__ == "__main__":
    from dataclasses import asdict

    print(asdict(config))
