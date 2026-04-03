__all__ = ["GraspWorkflow"]


def __getattr__(name):
    if name == "GraspWorkflow":
        from .workflow import GraspWorkflow

        return GraspWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
