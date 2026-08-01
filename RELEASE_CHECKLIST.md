# Publication release checklist

Use this checklist for the exact artifact cited by a manuscript.

- [ ] Replace the generic contributor entry in `CITATION.cff` with the authors.
- [ ] Create a clean Python environment from `requirements-lock.txt`.
- [ ] Run all automated tests successfully.
- [ ] Run `configs/publication_final.yaml` into a new empty directory.
- [ ] Confirm all 10 seeds completed with exactly 25,040 evaluations each.
- [ ] Confirm `publication_audit.json` reports `publication_checks_passed: true`.
- [ ] Confirm `run_manifest.json` reports `status: complete`.
- [ ] Inspect all seed-level voltage, SOC, power, and cycle-closure checks.
- [ ] Report the full seed distribution, not only the merged best design.
- [ ] Run `run_reliability_sensitivity.py` for the frozen selected design.
- [ ] Report deterministic/exponential repair and uncertainty sensitivities.
- [ ] Retain the paper AENS discrepancy (28.8063 calculated vs 24.48 reported).
- [ ] Do not describe energy adequacy as AC or transient-stability validation.
- [ ] Verify figure labels, units, captions, and significant digits manually.
- [ ] Archive source and outputs together; record the archive SHA-256.
- [ ] Deposit the archive in Zenodo or an institutional repository and add its DOI
      to `CITATION.cff` and the manuscript.

If any required constraint or provenance check fails, the numerical run is not
a publication result. An unsupported improvement hypothesis is not a failed
run; report it as an inconclusive or adverse result.
