# Lazy import to avoid dependency issues
def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as _run_pipeline
    return _run_pipeline(*args, **kwargs)
