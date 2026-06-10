from localllm.service.manager import ServiceManager

# NOTE: deliberately NOT re-exporting `app` here. `from localllm.service.app
# import app` would rebind the package attribute `app` from the submodule to
# the FastAPI instance, which breaks `unittest.mock.patch("localllm.service.
# app.X")` target resolution on Python 3.10 (it walks package attributes).
# Use `localllm.service.app:app` (uvicorn) or import the module directly.

__all__ = ["ServiceManager"]
