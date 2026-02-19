import sys
import os
import pytest
import asyncio
from src.mcp_server import mcp_server, setup_stdio_transport, register_tools, list_document_sections

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

def test_stdio_transport_setup():
    """Test if Stdio transport is set up correctly."""
    try:
        setup_stdio_transport()
        assert True  # If no exception, the setup is successful
    except Exception as e:
        pytest.fail(f"Stdio transport setup failed: {e}")

@pytest.mark.asyncio
async def test_tool_registration():
    """Test the registration of tools."""
    try:
        register_tools()
        tools = await mcp_server.list_tools()
        assert any(tool.name == "list_document_sections" for tool in tools)
    except Exception as e:
        pytest.fail(f"Tool registration failed: {e}")

def test_list_document_sections():
    """Test the functionality of the list_document_sections tool."""
    result = list_document_sections("doc123")
    assert result == {
        "document_id": "doc123",
        "sections": ["Introduction", "Chapter 1", "Conclusion"]
    }