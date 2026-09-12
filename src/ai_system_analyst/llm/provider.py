from abc import ABC, abstractmethod

from ai_system_analyst.llm.models import IncidentAnalysis


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
    ) -> IncidentAnalysis:
        """Generate a structured incident analysis."""
        raise NotImplementedError
