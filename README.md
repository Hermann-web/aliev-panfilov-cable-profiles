# Aliev–Panfilov cable profile counts

This directory supports one numerical claim in *Target Selection in Cardiac Inverse Problems from Non-Invasive Measurements* (ICESST 2026). The intended repository is <https://github.com/Hermann-web/aliev-panfilov-cable-profiles>. No DOI is assigned, and this checkout is not yet a published release.

## Claim these files support

On a one-dimensional Aliev–Panfilov cable, diffusivity is a unit background plus five cosine modes, with coefficients 0.20, −0.12, 0.08, −0.06, and 0.05. The observation is one remote pseudo-electrocardiogram. Additive Gaussian noise has standard deviation equal to 2%, 5%, 10%, or 20% of the root-mean-square of the noise-free trace, with one draw at each level. A coefficient is practically identifiable when both sides of its profile rise above 3.841459, the 95% point of the chi-square distribution with one degree of freedom.

Under that rule the five coefficients give 5/5, 1/5, 0/5, and 0/5. These counts are practical identifiability for this parameterization and this observation. They are not a failure theorem for an unrestricted diffusivity field.

The same rule, at 2% noise, leaves five of the six scalar preset parameters practically identifiable. `mu2` is one-sided, so that restricted profile fails the two-sided rule.

## Check the stored counts

From this directory:

```bash
python3 verify_profiles.py
```

The checker uses the Python standard library. It recounts every stored rise against the threshold, checks the four noise levels, and checks the file checksums in `PUBLIC-RELEASE-MANIFEST.json` and `SHA256SUMS`.

## Regenerate

`practical_identifiability.py` is the generator. NumPy is its only dependency. This checkout does not pin a NumPy build, so a fresh run is a recomputation rather than a byte-level replay of the released JSON.

```bash
python3 practical_identifiability.py --check
```

`--check` recomputes the six scalar profiles and compares their pass list with `practical-identifiability.json`. It does not rewrite that file.

`--scaling` and `--sweep` rewrite `identifiability-field-scaling.json` and `identifiability-crossover.json` beside the script. Copy the released JSON aside before using those modes. The field draw starts from seed `20260805 + n_modes`. The scalar draw uses seed `20260805`.

The released field-scaling file records the five cosine coefficients used for modes one to five. The generator also carries three further coefficients for the six-mode and seven-mode rows. At noise 0.02 those rows are 4/6 and 7/7. The six-mode margin is negative, and the mode sweep is not monotone. No optimizer convergence trace is stored.

## What this is not

The cable has one spatial dimension, one geometry, and one preset. The five-mode row does not rank inverse targets, and it does not establish non-identifiability of a general diffusivity field.

Rights and the publication block are in `RIGHTS.md`.
