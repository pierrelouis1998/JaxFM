from typing import Any

from jaxfm.flows.base import Flow
from jaxfm.flows.conditional_flows import CFM
from jaxfm.flows.reflows import ReFlow


def get_flow(name: str, model: Any, num_steps: int, **kwargs) -> Flow:
    """Get the flow model."""
    if name == "reflow":
        return ReFlow(model=model, num_steps=num_steps)
    elif name == "conditional":
        return CFM(model=model, num_steps=num_steps, **kwargs)
    else:
        raise ValueError(f"Unknown flow model {name}")
