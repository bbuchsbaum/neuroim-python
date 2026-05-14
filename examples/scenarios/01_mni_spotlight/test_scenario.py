"""Pointer file.

The runnable acceptance test for this scenario lives one level up at
``examples/scenarios/test_s01_mni_spotlight.py``.

That file is needed there because the leading digit in this folder's
name (``01_mni_spotlight``) makes it an illegal Python module, so
pytest cannot collect tests from inside it under the default
``prepend`` import mode.  The companion ``conftest.py`` excludes
this file from collection.

This file is kept only so the scenario folder remains a self-contained
reading unit (README + baseline + neuroim_version + REPORT + this
pointer).  See the up-level test file for the runnable assertions.
"""
