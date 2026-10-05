"""Declares the SegFormer runtime dependencies for the method installer."""

import Framework


__extension_name__ = "SegFormer runtime dependencies"
__install_command__ = [
    "pip", "install",
    "transformers==4.53.2",
    "accelerate==1.9.0",
]

try:
    import transformers as _transformers  # noqa: F401
    import accelerate as _accelerate  # noqa: F401
except ImportError as exc:
    raise Framework.ExtensionError(
        name=__extension_name__,
        install_command=__install_command__,
    ) from exc
