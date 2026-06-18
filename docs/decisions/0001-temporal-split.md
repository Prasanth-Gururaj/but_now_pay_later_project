# ADR 0001: Temporal Train/Test Split

## Status

Accepted

## Context

Standard random train/test splits leak future information into training data for time-series credit risk data. The LendingClub dataset spans 2007-2018 with `issue_d` as the temporal axis.

## Decision

Use a temporal split strategy:
- **Train**: 2013-2015
- **Validation**: 2016
- **Test**: 2017
- **Drift simulation (slight)**: 2018
- **Drift simulation (crisis)**: 2008-2009

## Consequences

- Prevents temporal data leakage
- Validation/test performance reflects realistic deployment scenarios
- 2018 and 2008-2009 windows enable drift detection testing without synthetic data
