"""Custom Sphinx extension for autodoc filtering."""


def skip_member(app, what, name, obj, skip, options):  # noqa: ARG001
    """Skip private members and problematic imports during autodoc.

    This filter helps keep documentation clean by excluding:
    - Private members (starting with _)
    - Implementation details
    - Auto-imported dependencies
    """
    # Skip private members
    if name.startswith("_"):
        return True

    # Skip specific problematic imports and base classes that clutter docs
    skip_names = {
        "BaseModel",  # Pydantic base
        "model_fields",  # Pydantic metadata
        "model_config",  # Pydantic configuration
        "ConfigDict",  # Pydantic config
    }

    if name in skip_names:
        return True

    return skip


def setup(app):
    """Setup the extension."""
    app.connect("autodoc-skip-member", skip_member)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
