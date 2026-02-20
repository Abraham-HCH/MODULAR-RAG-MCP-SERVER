"""Factory for creating Splitter instances.

This module implements the Factory Pattern to instantiate the appropriate
Splitter provider based on configuration, enabling configuration-driven selection
of different splitting strategies without code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from enum import Enum

from src.libs.splitter.base_splitter import BaseSplitter

class DefaultSplitter(BaseSplitter):
    """A basic splitter implementation for default use."""
    def __init__(self, settings: Settings, **kwargs: Any):
        self.settings = settings
        self.kwargs = kwargs

    def split_text(self, text: str, **kwargs: Any) -> list[str]:
        return text.splitlines()

if TYPE_CHECKING:
    from src.core.settings import Settings


class SplitterType(Enum):
    DEFAULT = "default"
    RECURSIVE = "recursive"
    FAKE = "fake"  # For testing purposes

# Export SplitterType for external use
__all__ = ["SplitterFactory", "SplitterType"]


def _register_builtin_providers() -> None:
    """Register built-in splitter providers.
    
    This function is called automatically when the module is imported.
    It registers all available splitter implementations with the factory.
    """
    # Import here to avoid circular imports and handle missing dependencies gracefully
    # Register default provider first
    try:
        SplitterFactory.register_provider("default", DefaultSplitter)
    except Exception:
        # ignore if already registered or other issues
        pass

    # Try to register recursive provider if available
    try:
        from src.libs.splitter.recursive_splitter import RecursiveSplitter
        SplitterFactory.register_provider("recursive", RecursiveSplitter)
    except ImportError:
        # langchain-text-splitters may be missing; recursive provider is optional
        pass


class SplitterFactory:
    """Factory for creating Splitter provider instances.
    
    This factory reads the splitter configuration from settings and instantiates
    the corresponding Splitter implementation. Supported providers will be added
    in subsequent tasks (B7.5).
    
    Design Principles Applied:
    - Factory Pattern: Centralizes object creation logic.
    - Config-Driven: Provider selection based on settings.yaml.
    - Fail-Fast: Raises clear errors for unknown providers.
    """
    
    _PROVIDERS: dict[str, type[BaseSplitter]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseSplitter]) -> None:
        """Register a new Splitter provider implementation.
        
        Args:
            name: The provider identifier (e.g., 'recursive', 'semantic', 'fixed').
            provider_class: The BaseSplitter subclass implementing the provider.
        
        Raises:
            ValueError: If provider_class doesn't inherit from BaseSplitter.
        """
        if not issubclass(provider_class, BaseSplitter):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseSplitter"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, splitter_type_or_settings=None, settings: Optional["Settings"] = None, splitter_type: Optional[object] = None, **override_kwargs: Any) -> BaseSplitter:
        """Create a Splitter instance based on configuration.

        Args:
            splitter_type: The type of splitter to create.
            settings: The application settings containing ingestion configuration.
            **override_kwargs: Optional parameters to override config values.

        Returns:
            An instance of the configured Splitter provider.

        Raises:
            ValueError: If the configured provider is not supported or missing.
        """
        # Support multiple call styles for compatibility:
        # - create(splitter_type: SplitterType, settings: Settings, **overrides)
        # - create("default", settings=Settings(), **overrides)
        # - create(settings: Settings, **overrides) where settings.ingestion.splitter defines the provider
        if splitter_type is not None:
            # explicit kwarg has highest priority
            provider_name = splitter_type.value.lower() if isinstance(splitter_type, SplitterType) else str(splitter_type).lower()
        elif isinstance(splitter_type_or_settings, SplitterType):
            provider_name = splitter_type_or_settings.value.lower()
        elif isinstance(splitter_type_or_settings, str):
            provider_name = splitter_type_or_settings.lower()
        else:
            # Called with settings as first arg
            settings = splitter_type_or_settings
            try:
                provider_str = settings.ingestion.splitter
            except Exception:
                raise ValueError(
                    "Missing ingestion configuration in settings. Expected settings.ingestion.splitter"
                )

            provider_name = provider_str.lower() if isinstance(provider_str, str) else str(provider_str).lower()
        provider_class = cls._PROVIDERS.get(provider_name)
        # If provider is not found and registry is empty, attempt to auto-register built-ins
        if provider_class is None and not cls._PROVIDERS:
            try:
                _register_builtin_providers()
            except Exception:
                # ignore errors during auto-registration; we'll raise below
                pass
            provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Splitter provider: '{provider_name}'. "
                f"Available providers: {available}. "
                "Provider implementations will be added in task B7.5."
            )

        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Splitter provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.
        
        Returns:
            Sorted list of available provider identifiers.
        """
        return sorted(cls._PROVIDERS.keys())


# Auto-register built-in providers when module is imported
_register_builtin_providers()

# Register DefaultSplitter as the default provider
SplitterFactory.register_provider("default", DefaultSplitter)