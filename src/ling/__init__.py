"""LP4FM spectral cluster-transition experiment — GPU-side foundation.

IMPORTANT IMPORT ORDER
----------------------
`ling.determinism` sets ``CUBLAS_WORKSPACE_CONFIG`` in ``os.environ`` at *import*
time. That environment variable is read by cuBLAS when the CUDA context is first
created, so it MUST be in the environment before torch initialises CUDA.

Importing this package imports ``ling.determinism`` first, so::

    import ling                    # or: from ling.determinism import enforce_determinism
    from ling.model import load_model

is safe. Importing torch and touching a CUDA tensor *before* importing
``ling.determinism`` is NOT safe; ``enforce_determinism()`` detects that case and
raises rather than silently running non-deterministically.
"""

from . import determinism as determinism  # noqa: F401  (side effect: sets CUBLAS env var)

__all__ = ["determinism"]
