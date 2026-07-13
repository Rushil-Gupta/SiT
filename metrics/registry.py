_EXTRACTORS = {}


def register(name):
    def decorator(cls):
        _EXTRACTORS[name] = cls
        return cls
    return decorator


def get_extractor(name, device="cuda", **kwargs):
    if name not in _EXTRACTORS:
        available = list(_EXTRACTORS.keys())
        raise ValueError(f"Unknown extractor '{name}'. Available: {available}")
    return _EXTRACTORS[name](device=device, **kwargs)


def list_extractors():
    return list(_EXTRACTORS.keys())
