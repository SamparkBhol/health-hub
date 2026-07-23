# Authorised aggregate surveillance input

Place the authorised, no-PII district-week export in this directory as
`district_week.csv`. Start from
[the versioned template](../templates/authorised_surveillance_district_week.csv).

Do not place patient records, names, addresses, mobile numbers, facility line
lists, clinical free text or credentials in this repository. The import contract
accepts only district/disease/week aggregate counts, completeness, population,
knowledge-vintage and threshold-version fields.

Validate an export before any model run:

```bash
uv run python scripts/audit_authorised_surveillance.py data/authorised_surveillance/district_week.csv
```

When the structural audit passes, build the offline disease/horizon artefact:

```bash
uv run python scripts/train_authorised_forecast.py \
  data/authorised_surveillance/district_week.csv
```

The application then serves the latest official observed-rate layer at
`/api/v1/observed-surveillance/map`, the model register at
`/api/v1/forecast/operational`, and only qualified current probabilities at
`/api/v1/forecast/operational/map`. Missing rows and failed model gates remain
typed missing/refusal states; neither becomes a zero or a fallback probability.
