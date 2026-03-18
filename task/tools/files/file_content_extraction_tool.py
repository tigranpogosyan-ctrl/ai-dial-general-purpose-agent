import json
from typing import Any

from aidial_sdk.chat_completion import Message

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.dial_file_conent_extractor import DialFileContentExtractor


class FileContentExtractionTool(BaseTool):
    """Tool to extract text content from files (PDF, TXT, CSV, HTML/HTM) with optional pagination."""

    PAGE_SIZE = 10_000  # Number of characters per page for large files

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    # -------------------------
    # Metadata
    # -------------------------

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "file_content_extraction_tool"

    @property
    def description(self) -> str:
        return (
            "Extracts text content from files. Supported: PDF (text only), TXT, CSV (as markdown table), HTML/HTM. "
            "PAGINATION: Files >10,000 chars are paginated. Response format: "
            "`**Page #X. Total pages: Y**` appears at end if paginated. "
            "USAGE: Start with page=1. If paginated, call again with page=2, page=3, etc. "
            "Always check response end for pagination info before answering user queries about file content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_url": {"type": "string", "description": "File URL"},
                "page": {
                    "type": "integer",
                    "description": "For large documents, each page consists of 10,000 characters.",
                    "default": 1,
                },
            },
            "required": ["file_url"],
        }

    # -------------------------
    # Execution
    # -------------------------

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # Parse arguments
        args = json.loads(tool_call_params.tool_call.function.arguments)
        file_url: str = args["file_url"]
        page: int = max(args.get("page", 1), 1)  # Ensure page >= 1

        stage = tool_call_params.stage
        stage.append_content("## Request arguments:\n")
        stage.append_content(f"**File URL**: {file_url}\n")
        if page > 1:
            stage.append_content(f"**Page**: {page}\n")
        stage.append_content("## Response:\n")

        # Extract text from file
        extractor = DialFileContentExtractor(endpoint=self.endpoint, api_key=tool_call_params.api_key)
        content = extractor.extract_text(file_url)

        if not content:
            content = "Error: File content not found."
            stage.append_content(content)
            return content

        # Handle pagination
        total_pages = (len(content) + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        if page > total_pages:
            return f"Error: Page {page} does not exist. Total pages: {total_pages}"

        start_idx = (page - 1) * self.PAGE_SIZE
        end_idx = start_idx + self.PAGE_SIZE
        page_content = content[start_idx:end_idx]

        final_content = f"{page_content}\n\n**Page #{page}. Total pages: {total_pages}**" if total_pages > 1 else page_content
        stage.append_content(f"```text\n{final_content}\n```\n")

        return final_content