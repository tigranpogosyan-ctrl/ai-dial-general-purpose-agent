import json
from typing import Any, Tuple, List

import faiss
import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.utils.dial_file_conent_extractor import DialFileContentExtractor


_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided document context.

- Use ONLY the provided context
- If insufficient info → say so clearly
- Be concise and direct
"""


class RagTool(BaseTool):

    def __init__(
        self,
        endpoint: str,
        deployment_name: str,
        document_cache: DocumentCache,
        model: SentenceTransformer | None = None,
        text_splitter: RecursiveCharacterTextSplitter | None = None,
    ):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.document_cache = document_cache

        # Inject dependencies (testable + reusable)
        self.model = model or SentenceTransformer("all-MiniLM-L6-v2")
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.file_extractor = DialFileContentExtractor(endpoint, api_key=None)

    # -------------------------
    # Metadata
    # -------------------------

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "rag_tool"

    @property
    def description(self) -> str:
        return (
            "Semantic document search tool. "
            "Use for answering questions about document content (PDF, TXT, CSV, HTML)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {"type": "string"},
                "file_url": {"type": "string"},
            },
            "required": ["request", "file_url"],
        }

    # -------------------------
    # Execution
    # -------------------------

    async def _execute(self, params: ToolCallParams) -> str | Message:
        args = self._parse_args(params)
        request = args["request"]
        file_url = args["file_url"]

        self._log_request(params, request, file_url)

        index, chunks = await self._get_or_create_index(params, file_url)

        if not chunks:
            return self._fail(params, "File content not found")

        retrieved_chunks = self._search(index, chunks, request)
        prompt = self._build_prompt(request, retrieved_chunks)

        self._log_prompt(params, prompt)

        return await self._generate_answer(params, prompt)

    # -------------------------
    # Steps (SRP)
    # -------------------------

    def _parse_args(self, params: ToolCallParams) -> dict:
        return json.loads(params.tool_call.function.arguments)

    def _log_request(self, params: ToolCallParams, request: str, file_url: str):
        stage = params.stage
        stage.append_content("## Request\n")
        stage.append_content(f"**Query**: {request}\n")
        stage.append_content(f"**File**: {file_url}\n")

    async def _get_or_create_index(
        self, params: ToolCallParams, file_url: str
    ) -> Tuple[faiss.Index, List[str]]:

        cache_key = f"{params.conversation_id}:{file_url}"
        cached = self.document_cache.get(cache_key)

        if cached:
            return cached

        extractor = DialFileContentExtractor(self.endpoint, params.api_key)
        text = extractor.extract_text(file_url)

        if not text:
            return None, []

        chunks = self.text_splitter.split_text(text)
        embeddings = self.model.encode(chunks).astype("float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        self.document_cache.set(cache_key, index, chunks)
        return index, chunks

    def _search(
        self, index: faiss.Index, chunks: List[str], query: str
    ) -> List[str]:
        query_embedding = self.model.encode([query]).astype("float32")

        k = min(3, len(chunks))
        _, indices = index.search(query_embedding, k)

        return [chunks[i] for i in indices[0]]

    def _build_prompt(self, request: str, chunks: List[str]) -> str:
        context = "\n\n".join(chunks)
        return f"CONTEXT:\n{context}\n---\nREQUEST: {request}"

    def _log_prompt(self, params: ToolCallParams, prompt: str):
        params.stage.append_content("## RAG Prompt\n```text\n")
        params.stage.append_content(prompt)
        params.stage.append_content("\n```\n")

    async def _generate_answer(self, params: ToolCallParams, prompt: str) -> str:
        dial = AsyncDial(base_url=self.endpoint, api_key=params.api_key)

        stream = await dial.chat.completions.create(
            messages=[
                {"role": Role.SYSTEM, "content": _SYSTEM_PROMPT},
                {"role": Role.USER, "content": prompt},
            ],
            deployment_name=self.deployment_name,
            stream=True,
        )

        content = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                params.stage.append_content(delta.content)
                content += delta.content

        return content

    def _fail(self, params: ToolCallParams, message: str) -> str:
        params.stage.append_content(f"## Error\n{message}\n")
        return message