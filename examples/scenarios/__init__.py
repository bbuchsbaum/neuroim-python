"""Comparison scenarios: neuroim public API vs raw nibabel+numpy.

Each subfolder ships a single neuroimaging task implemented twice — once
in canonical raw-nibabel+numpy form, and once through the neuroim public
API — together with a runnable parity test and a verdict report.

The scenarios are ordered from simplest to most complex.  The goal is to
*win* the comparison on line count and read-time, or at worst tie, and
to surface pain points in neuroim honestly when the win is narrower
than the mission would like.
"""
