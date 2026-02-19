"""
Splitter Integration Module

This module integrates the Splitter functionality into the Ingestion Pipeline.
"""

from src.libs.splitter.splitter_factory import SplitterFactory
from src.core.types import Document, Chunk

class SplitterIntegration:
    """
    Handles the integration of Splitter into the Ingestion Pipeline.
    """

    def __init__(self, splitter_type: str = "default", settings=None):
        """
        Initialize the SplitterIntegration with a specific splitter type.

        :param splitter_type: The type of splitter to use (default: "default").
        :param settings: Optional settings for the splitter.
        """
        self.splitter = SplitterFactory.create(settings=settings, splitter_type=splitter_type)

    def process_document(self, document: Document) -> list[Chunk]:
        """
        Process a document and split it into chunks.

        :param document: The document to be split.
        :return: A list of chunks generated from the document.
        """
        return [Chunk(id=f"chunk_{i}", text=chunk, metadata=document.metadata) for i, chunk in enumerate(self.splitter.split_text(document.text))]

    def list_available_splitters(self) -> list[str]:
        """
        List all available splitter providers.

        :return: A list of available splitter provider names.
        """
        return SplitterFactory.list_providers()

# Example usage
if __name__ == "__main__":
    # Example document
    example_document = Document(
        text="""# Title\n\nThis is a paragraph.\n\n## Subtitle\n\nAnother paragraph.""",
        metadata={"source": "example.pdf"}
    )

    # Initialize the integration
    splitter_integration = SplitterIntegration(splitter_type="recursive")

    # List available splitters
    available_splitters = splitter_integration.list_available_splitters()
    print("Available Splitters:", available_splitters)

    # Process the document
    chunks = splitter_integration.process_document(example_document)

    # Output the chunks
    for chunk in chunks:
        print(chunk)