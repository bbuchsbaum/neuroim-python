"""Pytest config for the scenarios suite.

Scenario folders are numbered (``01_mni_spotlight``, ``02_...``) so they
sort visually.  The leading digit makes them illegal Python module names,
so pytest cannot collect tests *inside* those folders under the default
``prepend`` import mode.

We resolve this by keeping the runnable acceptance test for each
scenario one level up — e.g. :mod:`scenarios.test_s01_mni_spotlight`
loads the digit-prefixed siblings via :mod:`importlib`.  The
``test_scenario.py`` file kept *inside* each scenario folder is for
inline reading only and is excluded from collection here.
"""

collect_ignore_glob = ["[0-9]*"]
