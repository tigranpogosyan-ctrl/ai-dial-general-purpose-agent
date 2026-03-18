from typing import Any

from aidial_sdk.chat_completion import Message
from pydantic import StrictStr

from task.tools.deployment.base import DeploymentTool
from task.tools.models import ToolCallParams


class ImageGenerationTool(DeploymentTool):
    """Generates images based on a text description using a deployed model."""

    # -------------------------
    # Tool Execution
    # -------------------------
    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # Call base deployment tool execution
        msg = await super()._execute(tool_call_params)

        # Append image markdown URLs to choice content if available
        if msg.custom_content and msg.custom_content.attachments:
            for attachment in msg.custom_content.attachments:
                if attachment.type in ("image/png", "image/jpeg"):
                    tool_call_params.choice.append_content(f"\n![image]({attachment.url})\n")

            # Provide fallback content if none exists
            if not msg.content:
                msg.content = StrictStr(
                    "The image has been successfully generated and displayed to the user."
                )

        return msg

    # -------------------------
    # Metadata
    # -------------------------
    @property
    def deployment_name(self) -> str:
        return "dall-e-3"

    @property
    def name(self) -> str:
        return "image_generation_tool"

    @property
    def description(self) -> str:
        return (
            "# Image Generator\n"
            "Generates images based on a text description.\n\n"
            "## Instructions:\n"
            "- Use this tool when the user asks to generate an image from a description or visualize information.\n"
            "- Select the best size based on user request; if a specific size is requested, pick the closest supported option.\n"
            "- Always include the markdown image URL in the response, followed by a brief description.\n\n"
            "## Restrictions:\n"
            "- Do not use this tool for data or numerical visualization."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Extensive description of the image that should be generated."
                },
                "size": {
                    "type": "string",
                    "description": "The size of the generated image.",
                    "enum": ["1024x1024", "1024x1792", "1792x1024"],
                    "default": "1024x1024"
                },
                "style": {
                    "type": "string",
                    "description": (
                        "Style of the generated image: `vivid` for hyperrealistic/dramatic images, "
                        "`natural` for more realistic/less dramatic images."
                    ),
                    "enum": ["natural", "vivid"],
                    "default": "natural"
                },
                "quality": {
                    "type": "string",
                    "description": "Image quality. `hd` generates images with finer details and consistency.",
                    "enum": ["standard", "hd"],
                    "default": "standard"
                },
            },
            "required": ["prompt"]
        }