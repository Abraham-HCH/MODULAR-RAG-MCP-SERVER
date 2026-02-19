class StdioTransport:
    """Placeholder implementation for StdioTransport."""

    def __init__(self):
        print("StdioTransport initialized")

    def send(self, message: str):
        """Simulate sending a message."""
        print(f"Sending message: {message}")

    def receive(self) -> str:
        """Simulate receiving a message."""
        return input("Enter a message: ")