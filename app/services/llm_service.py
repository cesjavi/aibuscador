import logging
import httpx

from app.config import get_settings


INSUFFICIENT_INFORMATION = "No hay información suficiente en los documentos cargados."
logger = logging.getLogger("rag.llm")


class LLMService:
    """Provider adapter for OpenAI, Groq, Ollama and LM Studio chat APIs."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def generate(self, prompt: str) -> str:
        provider = self.settings.llm_provider.lower()
        logger.info(
            "LLM request provider=%s model=%s prompt_chars=%s prompt_preview=%r",
            provider,
            self.settings.model_name,
            len(prompt),
            prompt[:1200],
        )
        if provider == "openai":
            return await self._openai_compatible(
                base_url="https://api.openai.com/v1",
                api_key=self.settings.openai_api_key,
                model=self.settings.model_name,
                prompt=prompt,
            )
        if provider == "groq":
            return await self._openai_compatible(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.settings.groq_api_key,
                model=self.settings.model_name,
                prompt=prompt,
            )
        if provider == "lmstudio":
            return await self._openai_compatible(
                base_url=self.settings.lmstudio_base_url.rstrip("/"),
                api_key="lm-studio",
                model=self.settings.model_name,
                prompt=prompt,
            )
        if provider == "ollama":
            return await self._ollama(prompt)
        raise ValueError(f"Proveedor LLM no soportado: {provider}")

    async def _openai_compatible(self, base_url: str, api_key: str | None, model: str, prompt: str) -> str:
        if not api_key:
            raise ValueError("Falta configurar la API key del proveedor LLM.")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._system_message()},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

    async def _ollama(self, prompt: str) -> str:
        payload = {
            "model": self.settings.model_name,
            "messages": [
                {"role": "system", "content": self._system_message()},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{self.settings.ollama_base_url.rstrip('/')}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"].strip()

    @staticmethod
    def _system_message() -> str:
        return (
            "Sos un asistente RAG. Respondé únicamente con la información del contexto provisto. "
            "Si el contexto contiene código, SQL, nombres de archivos o ejemplos operativos relevantes, "
            "podés explicar el patrón observado, proponer pasos basados en esos ejemplos y aclarar qué parte "
            "no está confirmada por los documentos. "
            f"Si no hay ningún contexto relevante, respondé exactamente: {INSUFFICIENT_INFORMATION}"
        )
