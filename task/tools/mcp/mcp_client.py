from typing import Optional, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import (
    CallToolResult,
    TextContent,
    ReadResourceResult,
    TextResourceContents,
    BlobResourceContents,
)
from pydantic import AnyUrl

from task.tools.mcp.mcp_tool_model import MCPToolModel


class MCPClient:
    """Handles MCP server connection and tool execution"""

    def __init__(self, mcp_server_url: str) -> None:
        self.server_url = mcp_server_url
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None

    # -------------------------
    # Factory
    # -------------------------

    @classmethod
    async def create(cls, mcp_server_url: str) -> "MCPClient":
        instance = cls(mcp_server_url)
        await instance.connect()
        return instance

    # -------------------------
    # Connection
    # -------------------------

    async def connect(self) -> None:
        """Connect to MCP server"""
        if self.session is not None:
            return

        try:
            # Streams context
            self._streams_context = streamablehttp_client(self.server_url)
            read_stream, write_stream, _ = await self._streams_context.__aenter__()

            # Session context
            self._session_context = ClientSession(read_stream, write_stream)
            self.session = await self._session_context.__aenter__()

            # Initialize
            await self.session.initialize()
            await self.session.send_ping()

        except Exception as e:
            await self.close()
            raise RuntimeError(f"Failed to connect to MCP server: {e}") from e

    def _ensure_connected(self) -> ClientSession:
        if not self.session:
            raise RuntimeError("MCP client not connected.")
        return self.session

    # -------------------------
    # Tools
    # -------------------------

    async def get_tools(self) -> list[MCPToolModel]:
        session = self._ensure_connected()

        tools = await session.list_tools()
        return [
            MCPToolModel(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
            )
            for tool in tools.tools
        ]

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        session = self._ensure_connected()

        result: CallToolResult = await session.call_tool(tool_name, tool_args)

        if not result.content:
            return None

        content = result.content[0]

        if isinstance(content, TextContent):
            return content.text

        return content  # fallback (binary / structured)

    # -------------------------
    # Resources
    # -------------------------

    async def get_resource(self, uri: AnyUrl) -> str | bytes:
        session = self._ensure_connected()

        result: ReadResourceResult = await session.read_resource(uri)

        if not result.contents:
            raise ValueError(f"No content in resource: {uri}")

        content = result.contents[0]

        if isinstance(content, TextResourceContents):
            return content.text

        if isinstance(content, BlobResourceContents):
            return content.blob

        raise TypeError(f"Unexpected resource content type: {type(content)}")

    # -------------------------
    # Cleanup
    # -------------------------

    async def close(self) -> None:
        """Close connection safely"""

        # Close session first
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception:
                pass

        # Then close streams
        if self._streams_context:
            try:
                await self._streams_context.__aexit__(None, None, None)
            except Exception:
                pass

        # Reset state
        self.session = None
        self._session_context = None
        self._streams_context = None

    # -------------------------
    # Context manager
    # -------------------------

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        await self.close()
        return False