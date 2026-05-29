"""Defense ????"""

from .miashield import MIAShieldDefense
from .hamp import HAMPDefense
from .query_sanitization import QuerySanitizationDefense
from .epd import EPDDefense
from .pii_output_filter import PIIOutputFilterDefense


DEFENSE_REGISTRY = {}


def register_defense(name, cls):
    """?? defense ??"""
    DEFENSE_REGISTRY[name] = cls


def get_defense(name):
    """????? defense ??"""
    if name not in DEFENSE_REGISTRY:
        available = ", ".join(sorted(DEFENSE_REGISTRY))
        raise ValueError(f"Unknown defense: {name}. Available: {available}")
    return DEFENSE_REGISTRY[name]


register_defense("miashield", MIAShieldDefense)
register_defense("hamp", HAMPDefense)
register_defense("query_sanitization", QuerySanitizationDefense)
register_defense("epd", EPDDefense)
register_defense("pii_output_filter", PIIOutputFilterDefense)
