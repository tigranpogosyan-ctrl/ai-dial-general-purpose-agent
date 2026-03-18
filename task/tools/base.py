from abc import ABC, abstractmethod
from typing import Any, Union

from aidial_client.types.chat import ToolParam, FunctionParam
from aidial_client.types.chat.legacy.chat_completion import Role
from aidial_sdk.chat_completion import Message
from pydantic import StrictStr

from task.tools.models import ToolCallParams


class BaseTool(ABC):
    """
    Base class for all tools.
    Provides unified execution flow, error handling, and schema generation.
    """

    async def execute(self, tool_call_params: ToolCallParams) -> Message:
        """
        Public execution wrapper.
        Handles:
        - message creation
        - error handling
        - result normalization
        """
        base_msg = self._create_base_message(tool_call_params)

        try:
            result = await self._execute(tool_call_params)
            return self._normalize_result(result, base_msg)

        except Exception as e:
            return self._handle_error(e, base_msg)

    # -------------------------
    # Internal helpers
    # -------------------------

    def _create_base_message(self, tool_call_params: ToolCallParams) -> Message:
        return Message(
            role=Role.TOOL,
            name=StrictStr(tool_call_params.tool_call.function.name),
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
        )

    def _normalize_result(
        self,
        result: Union[str, Message],
        base_msg: Message
    ) -> Message:
        if isinstance(result, Message):
            return result

        base_msg.content = StrictStr(str(result))
        return base_msg

    def _handle_error(self, error: Exception, base_msg: Message) -> Message:
        # In production: replace with logging
        base_msg.content = StrictStr(
            f"[ERROR] Tool execution failed: {type(error).__name__}: {error}"
        )
        return base_msg

    # -------------------------
    # Abstract API
    # -------------------------

    @abstractmethod
    async def _execute(self, tool_call_params: ToolCallParams) -> Union[str, Message]:
        """Core business logic of the tool."""
        pass

    @property
    def show_in_stage(self) -> bool:
        return True

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON schema for tool parameters."""
        pass

    # -------------------------
    # Schema
    # -------------------------

    @property
    def schema(self) -> ToolParam:
        return ToolParam(
            type="function",
            function=FunctionParam(
                name=self.name,
                description=self.description,
                parameters=self.parameters,
            ),
        )