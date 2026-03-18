import io
from pathlib import Path
from typing import Callable, Dict

import pdfplumber
import pandas as pd
from aidial_client import Dial
from bs4 import BeautifulSoup


class DialFileContentExtractor:
    def __init__(self, endpoint: str, api_key: str):
        self.dial_client = Dial(base_url=endpoint, api_key=api_key)

        # Strategy pattern: map extensions to handlers
        self._handlers: Dict[str, Callable[[bytes], str]] = {
            ".txt": self._handle_txt,
            ".pdf": self._handle_pdf,
            ".csv": self._handle_csv,
            ".html": self._handle_html,
            ".htm": self._handle_html,
        }

    def extract_text(self, file_url: str) -> str:
        response = self.dial_client.files.download(file_url)

        file_content = response.get_content()
        file_extension = Path(response.filename).suffix.lower()

        handler = self._handlers.get(file_extension, self._handle_default)
        return self._safe_execute(handler, file_content, response.filename)

    # -------------------------
    # Handlers
    # -------------------------

    def _handle_txt(self, content: bytes) -> str:
        return content.decode("utf-8", errors="ignore")

    def _handle_pdf(self, content: bytes) -> str:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()

    def _handle_csv(self, content: bytes) -> str:
        df = pd.read_csv(io.StringIO(content.decode("utf-8", errors="ignore")))
        return df.to_markdown(index=False)

    def _handle_html(self, content: bytes) -> str:
        soup = BeautifulSoup(content, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        return soup.get_text(separator="\n", strip=True)

    def _handle_default(self, content: bytes) -> str:
        return content.decode("utf-8", errors="ignore")

    # -------------------------
    # Error handling
    # -------------------------

    def _safe_execute(self, handler: Callable[[bytes], str], content: bytes, filename: str) -> str:
        try:
            return handler(content)
        except Exception as e:
            print(f"[ERROR] Failed to process file '{filename}': {e}")
            return ""