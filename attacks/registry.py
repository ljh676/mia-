"""攻击方法注册器。

新增攻击方法时，只需要：
1. 在 attacks/ 下新增一个包装类，继承 BaseAttack。
2. 在 ATTACK_REGISTRY 中注册名称和类。
3. 准备对应配置文件。
"""

from typing import Any, Dict

from .dcmi import DCMIAttack
from .rag_mia import RAGMIAAttack
from .neighbour_mia import NeighbourMIAAttack
from .spv_mia import SPVMIAAttack
from .janus import JanusAttack
from .janus_mia import JanusMIAAttack
from .janus_lora_transfer import JanusLoRATransferAttack


ATTACK_REGISTRY = {
    "rag_mia": RAGMIAAttack,
    "dcmi": DCMIAttack,
    "neighbour_mia": NeighbourMIAAttack,
    "spv_mia": SPVMIAAttack,
    "janus": JanusAttack,
    "janus_mia": JanusMIAAttack,
    "janus_lora_transfer": JanusLoRATransferAttack,
}


def get_attack(attack_name: str, config: Dict[str, Any]):
    """根据攻击名称返回攻击实例。

    Args:
        attack_name: 攻击方法名称，例如 rag_mia、dcmi。
        config: 攻击方法配置。

    Returns:
        BaseAttack 子类实例。

    Raises:
        ValueError: 当攻击名称未注册时抛出。
    """
    normalized_name = attack_name.lower()
    if normalized_name not in ATTACK_REGISTRY:
        available = ", ".join(sorted(ATTACK_REGISTRY))
        raise ValueError(f"未知攻击方法: {attack_name}，当前可用: {available}")
    return ATTACK_REGISTRY[normalized_name](config)
