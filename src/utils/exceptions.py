class APIKeyNotFoundError(Exception):
    """
    Raised when the API key is not defined/declared.

    Args:
        Exception (Exception): APIKeyNotFoundError
    """


class MethodNotImplementedError(Exception):
    """
    Raised when a method is not implemented.

    Args:
        Exception (Exception): MethodNotImplementedError
    """


class UnsupportedOpenAIModelError(Exception):
    """
    Raised when an unsupported OpenAI model is used.

    Args:
        Exception (Exception): UnsupportedOpenAIModelError
    """