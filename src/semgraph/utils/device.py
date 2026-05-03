import torch


def _get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device(device: str | None = "auto") -> torch.device:
    if device == "auto" or device is None:
        return _get_default_device()
    else:
        return torch.device(device)
