from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from ..schemas import Evidence


class SemanticAnalyzer(ABC):
    @abstractmethod
    async def analyze(
        self,
        speaker: str,
        text: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Evidence:
        raise NotImplementedError
