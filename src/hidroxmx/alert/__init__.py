"""Auditable Mamdani fuzzy alert layer + rule export.

Module to be filled in Milestone 5 (§12.5) :

- ``fuzzy.py`` — Mamdani FIS mapping (forecast percentile, interval width /
  exceedance probability) to alert class (none / watch / warning / emergency);
  membership functions anchored to local action thresholds (co-designed with
  CONAGUA / Protección Civil).
- ``rules.py`` — export the IF–THEN rule table in a human-readable file for
  auditability and for the paper's supplementary material.
"""
