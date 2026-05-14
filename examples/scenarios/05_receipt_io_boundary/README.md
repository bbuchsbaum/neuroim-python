# Scenario 05 — receipts across the IO boundary

Write a derived map (a searchlight statistic) to disk, then in a
**fresh** Python process — with no neuroim import context and no
upstream Python objects — answer the question:

> *What produced this .nii.gz, from what input volume, with what mask
> and what method?*

This is the realistic collaborator-handoff scenario.  ME-9 made
receipts compose in memory; this scenario asks whether they survive
contact with the file system.

The acceptance test lives one level up at
`scenarios/test_s05_receipt_io_boundary.py`.  The report and pain
points are in `REPORT.md`.
