# Security policy

## Supported versions

Only the latest release receives security fixes.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
[Report a vulnerability](https://github.com/BenjaminLucier/predict/security/advisories/new).
If you can't use GitHub, email benjaminlucier@gmail.com instead. Please don't open a public
issue or PR for a suspected vulnerability.

You can expect an acknowledgement within 3 working days. Fixes are coordinated
with you, and details are disclosed publicly once a fixed release is available,
or after 90 days at the latest.

## Scope

In scope: the code in this repository, in particular

- panics, hangs or unbounded memory use on inputs the documentation says are
  accepted (for example a sparse matrix that passed validation);
- results that break a documented guarantee, such as a dataset in the
  `Finite` state holding NaN or infinity;
- undefined behaviour, in this crate or reached through its public API.

This crate reads no files, makes no network requests and holds no secrets.

Out of scope:

- Vulnerabilities in dependencies that don't affect this code. Report them
  upstream; fixes are picked up through Dependabot.
- Panics documented as such (an out-of-range row index, for example).
- The documentation site, predict.rs.

## Safe harbour

Good-faith research that follows this policy, avoids accessing other people's
data, and gives us reasonable time to fix issues will not be pursued by the
maintainers.
