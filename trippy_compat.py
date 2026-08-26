"""
Phase 4 support
trippy_compat.py: make TriPPy importable on Python 3.12+.

TriPPy still does `import imp`, but the stdlib `imp` module was removed in
Python 3.12. CANFAR's containers run 3.12 and 3.13, so `import trippy` fails with
"No module named 'imp'" before any of our code runs. Importing this module first
installs a minimal `imp` shim built on importlib, covering the handful of
functions old code uses. Idempotent: safe to import many times.

Usage: `import trippy_compat` before importing trippy anywhere.
"""

import io
import sys
import types
import contextlib
import importlib
import importlib.util
import importlib.machinery

import numpy as np


def install_imp_shim():
    """Register a minimal `imp` module if the real one is gone (py>=3.12)."""
    if "imp" in sys.modules:
        return
    try:
        import imp  # noqa: F401  (present on <3.12, nothing to do)
        return
    except ImportError:
        pass

    m = types.ModuleType("imp")

    def load_source(name, pathname, file=None):
        spec = importlib.util.spec_from_file_location(name, pathname)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    def find_module(name, path=None):
        spec = importlib.machinery.PathFinder().find_spec(name, path)
        if spec is None:
            raise ImportError(name)
        return None, getattr(spec, "origin", None), ("", "", 0)

    m.load_source = load_source
    m.find_module = find_module
    m.new_module = types.ModuleType
    m.reload = importlib.reload
    m.acquire_lock = m.release_lock = lambda: None
    m.lock_held = lambda: False
    m.PY_SOURCE, m.PY_COMPILED, m.C_EXTENSION, m.PKG_DIRECTORY = 1, 2, 3, 5
    sys.modules["imp"] = m


def patch_none_line(psf_module):
    """
    TriPPy's modelPSF restore rebuilds its lookup table by calling
    line(self.rate, self.angle, self.dt). For a plain (non-trailed) PSF, as the
    CLASSY .psf.fits files are, those come back None and line() crashes on
    `None * float`. Coerce None to 0 so a plain PSF restores as a zero-motion
    point source; the real trail is set by the explicit line() call that
    injection makes afterwards. Idempotent.
    """
    mp = getattr(psf_module, "modelPSF", None)
    if mp is None or getattr(mp, "_none_line_patched", False):
        return
    orig_line = mp.line
    DEFAULT_PIXSCALE = 0.185          # MegaCam arcsec/pixel, only used at rate 0

    def line(self, rate=0.0, angle=0.0, dt=0.0, *args, **kwargs):
        # a plain restored PSF has None for every trail field; coerce them so a
        # zero-motion (point) lookup table builds instead of crashing on None
        rate = 0.0 if rate is None else rate
        angle = 0.0 if angle is None else angle
        dt = 0.0 if dt is None else dt
        if not args and kwargs.get("pixScale", None) is None:
            kwargs["pixScale"] = getattr(self, "pixScale", None) or DEFAULT_PIXSCALE
        if getattr(self, "pixScale", None) is None:
            self.pixScale = kwargs.get("pixScale", DEFAULT_PIXSCALE) or DEFAULT_PIXSCALE
        return orig_line(self, rate, angle, dt, *args, **kwargs)

    mp.line = line
    mp._none_line_patched = True


def patch_plant_scalars(psf_module):
    """
    TriPPy's plant() does int(x) on its position argument. NumPy 2.0 forbids
    int() on a 1-element array, which is what the callers pass, so coerce a
    size-1 array to a Python scalar first. Multi-source arrays are left alone.
    Idempotent.
    """
    mp = getattr(psf_module, "modelPSF", None)
    if mp is None or getattr(mp, "_plant_scalar_patched", False):
        return
    orig_plant = mp.plant

    def _scalar(v):
        arr = np.asarray(v)
        return arr.item() if arr.size == 1 else v

    def plant(self, x, y, amp, *args, **kwargs):
        return orig_plant(self, _scalar(x), _scalar(y), _scalar(amp),
                          *args, **kwargs)

    mp.plant = plant
    mp._plant_scalar_patched = True


def patch_quiet(psf_module):
    """
    TriPPy prints "Restoring PSF...", "PSF restored." and "Using the lookup
    table..." on every restore and line() call, unconditionally. With one
    injection per frame per sample that floods training. Wrap the noisy methods
    to swallow their stdout. Idempotent.
    """
    mp = getattr(psf_module, "modelPSF", None)
    if mp is None or getattr(mp, "_quiet_patched", False):
        return

    def quiet(fn):
        def wrapped(*args, **kwargs):
            with contextlib.redirect_stdout(io.StringIO()):
                return fn(*args, **kwargs)
        return wrapped

    for name in ("line", "plant", "_fitsReStore"):
        if hasattr(mp, name):
            setattr(mp, name, quiet(getattr(mp, name)))
    mp._quiet_patched = True


def patch_trippy(psf_module):
    """Apply every TriPPy compatibility patch: None trail fields, NumPy-2 scalar
    planting, and silencing the per-call stdout chatter."""
    patch_none_line(psf_module)
    patch_plant_scalars(psf_module)
    patch_quiet(psf_module)


# install on import so a bare `import trippy_compat` is enough
install_imp_shim()


def _self_check():
    # imp shim is present
    import imp  # noqa: F401
    # patch coerces None on a fake modelPSF that mimics the crash
    import types

    class FakePSF:
        pixScale = None
        def line(self, rate, angle, dt, pixScale=0.2, *a, **k):
            # mimics trippy: divides by pixScale, so None rate or None pixScale crash
            return rate * 1.0 + angle * 1.0 + dt / pixScale
        def plant(self, x, y, amp, *a, **k):
            # mimics trippy: int(x) crashes on a 1-element numpy array under numpy 2
            print("chatter from plant")           # trippy prints on every call
            return int(x) + int(y) + amp

    mod = types.ModuleType("fake_trippy_psf")
    mod.modelPSF = FakePSF
    patch_trippy(mod)
    # the exact call trippy's restore makes: everything None, pixScale as kwarg
    assert FakePSF().line(None, None, None, pixScale=None) == 0.0, "None not coerced"
    # the exact call make_psf_stamp makes: 1-element arrays that int() rejects,
    # and the chatter must be swallowed
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        val = FakePSF().plant(np.array([5.0]), np.array([3.0]), np.array([1.0]))
    assert val == 9.0, "1-element array not coerced to scalar"
    assert "chatter" not in buf.getvalue(), "trippy chatter not silenced"
    patch_trippy(mod)                                 # idempotent, no double wrap
    print("trippy_compat self-check passed: imp shim, None-line, numpy-2 plant, quiet")


if __name__ == "__main__":
    _self_check()
