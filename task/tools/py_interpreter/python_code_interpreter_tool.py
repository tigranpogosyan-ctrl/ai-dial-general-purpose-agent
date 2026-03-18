import base64
import json
from typing import Any, Optional

from aidial_client import Dial
from aidial_sdk.chat_completion import Message, Attachment
from pydantic import StrictStr, AnyUrl

from task.tools.base import BaseTool
from task.tools.py_interpreter._response import _ExecutionResult
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


class PythonCodeInterpreterTool(BaseTool):

    def __init__(
        self,
        mcp_client: MCPClient,
        tool: MCPToolModel,
        dial_endpoint: str,
    ):
        self._mcp_client = mcp_client
        self._tool = tool
        self.dial_endpoint = dial_endpoint

    # -------------------------
    # Factory
    # -------------------------

    @classmethod
    async def create(
        cls,
        mcp_url: str,
        tool_name: str,
        dial_endpoint: str,
    ) -> "PythonCodeInterpreterTool":

        mcp_client = await MCPClient.create(mcp_url)
        tools = await mcp_client.get_tools()

        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in MCP")

        return cls(mcp_client, tool, dial_endpoint)

    # -------------------------
    # Metadata
    # -------------------------

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._tool.parameters

    # -------------------------
    # Execution
    # -------------------------

    async def _execute(self, params: ToolCallParams) -> str | Message:
        args = self._parse_args(params)

        self._log_request(params, args)

        result = await self._call_mcp(args)

        await self._handle_files(params, result)

        self._truncate_output(result)
        self._log_result(params, result)

        return StrictStr(result.model_dump_json())

    # -------------------------
    # Steps (SRP)
    # -------------------------

    def _parse_args(self, params: ToolCallParams) -> dict:
        return json.loads(params.tool_call.function.arguments)

    def _log_request(self, params: ToolCallParams, args: dict):
        stage = params.stage
        code = args["code"]
        session_id = args.get("session_id")

        stage.append_content("## Request\n")
        stage.append_content(f"```python\n{code}\n```\n")

        if session_id:
            stage.append_content(f"**session_id**: {session_id}\n")
        else:
            stage.append_content("New session will be created\n")

        stage.append_content("## Response\n")

    async def _call_mcp(self, args: dict) -> _ExecutionResult:
        raw = await self._mcp_client.call_tool(self.name, args)
        return _ExecutionResult.model_validate(json.loads(raw))

    async def _handle_files(self, params: ToolCallParams, result: _ExecutionResult):
        if not result.files:
            return

        dial = Dial(base_url=self.dial_endpoint, api_key=params.api_key)
        files_home = dial.my_appdata_home()

        for file in result.files:
            attachment = await self._process_file(file, dial, files_home)

            params.stage.add_attachment(attachment)
            params.choice.add_attachment(attachment)

        result.instructions = (
            "Files generated. DO NOT include file links in response."
        )

    async def _process_file(self, file, dial: Dial, files_home) -> Attachment:
        resource = await self._mcp_client.get_resource(AnyUrl(file.uri))

        file_data = self._decode_file(resource, file.mime_type)

        url = f"files/{(files_home / file.name).as_posix()}"
        dial.files.upload(url=url, file=file_data)

        return Attachment(
            url=StrictStr(url),
            type=StrictStr(file.mime_type),
            title=StrictStr(file.name),
        )

    def _decode_file(self, resource: str, mime_type: str) -> bytes:
        if mime_type.startswith("text/") or mime_type in {
            "application/json",
            "application/xml",
        }:
            return resource.encode("utf-8")

        return base64.b64decode(resource)

    def _truncate_output(self, result: _ExecutionResult):
        if result.output:
            result.output = [o[:200] for o in result.output]

    def _log_result(self, params: ToolCallParams, result: _ExecutionResult):
        params.stage.append_content("```json\n")
        params.stage.append_content(result.model_dump_json(indent=2))
        params.stage.append_content("\n```\n")