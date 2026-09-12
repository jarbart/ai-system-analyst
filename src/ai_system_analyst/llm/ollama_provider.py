import json
from urllib.request import Request, urlopen

from ai_system_analyst.llm.models import IncidentAnalysis
from ai_system_analyst.llm.provider import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = "qwen2.5:0.5b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    def generate(self, prompt: str) -> IncidentAnalysis:
        payload = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "format": IncidentAnalysis.model_json_schema(),
            }
        ).encode("utf-8")

        request = Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))

        return IncidentAnalysis.model_validate_json(data["response"])
