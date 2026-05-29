"""攻击方法统一抽象接口。"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAttack(ABC):
    """所有成员推断攻击方法都必须遵循的统一接口。

    设计目标：
    1. 不侵入原论文仓库，只在本框架中做适配和封装。
    2. 所有攻击方法都暴露 prepare / run / evaluate 三个阶段。
    3. 所有攻击方法最终都返回统一字典，便于后续横向比较。
    """

    def __init__(self, config: Dict[str, Any]):
        """保存攻击配置。

        Args:
            config: 从 yml 配置文件读取出的配置字典。
        """
        self.config = config
        self.result: Dict[str, Any] = {}

    @abstractmethod
    def prepare(self) -> None:
        """准备攻击运行所需资源，例如检查原项目路径、入口脚本、输出目录。"""
        raise NotImplementedError

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """执行攻击主流程，并返回至少包含 scores / predictions / labels 的结果。"""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """计算统一评价指标，并把 metrics 写回结果字典。"""
        raise NotImplementedError
