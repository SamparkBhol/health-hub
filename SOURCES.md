# Source, access and rights register

**Checked:** 22 July 2026  
**Implemented registry:** 170 routes; 156 enabled. Route-language memberships: 63 Odia, 46 Hindi and 48 English (a route can carry more than one language).  
**Rule:** HTTP reachability proves only technical access. It does not grant permission to retain, redistribute, republish or train on content.

The complete executable register is [config/sources.yaml](config/sources.yaml). The table below retains the most important government/core routes; it is not the complete 131-row inventory.

## Core acquisition routes

| ID | Canonical route | Language / format | Enabled | Recorded access and rights state |
|---|---|---|---:|---|
| `odisha_hfw_circulars_en` | [Odisha H&FW English circulars](https://health.odisha.gov.in/en/notifications/circulars) | EN; HTML index → PDF | Yes | Index reachable; `/robots.txt` returned 403, which is recorded as unavailable rather than interpreted as permission. Official-source review remains required before full-text redistribution. PDFs are receipt/hash-only until their exact digest is approved. |
| `odisha_hfw_circulars_or` | [Odisha H&FW Odia circulars](https://health.odisha.gov.in/or/notifications/circulars) | OR; HTML index → PDF | Yes | Same access/rights boundary as the English route. Some observed PDFs are raster scans with unusable embedded text. |
| `nhm_odisha_notifications` | [NHM Odisha notifications](https://nhmodisha.gov.in/notificationss/) | EN; HTML index → PDF | Yes | Reachable; observed robots policy disallowed `/wp-admin/`, not the registered path. Official-source review remains required before full-text redistribution. |
| `pib_bhubaneswar_en` | [PIB Bhubaneswar English](https://www.pib.gov.in/allRel.aspx?reg=21&lang=1) | EN; HTML index | No | The actual collector received HTTP 403 on 21 July 2026. The route remains registered with PIB's attribution policy but is unavailable to this collector. |
| `pib_bhubaneswar_hi` | [PIB Bhubaneswar Hindi](https://www.pib.gov.in/allRel.aspx?reg=21&lang=2) | HI; HTML index | No | The actual collector received HTTP 403 on 21 July 2026. It is not claimed as working Hindi coverage. |
| `newsonair_odisha_dengue_hi` | [Akashvani Hindi Odisha-dengue search](https://newsonair.gov.in/hi/?s=%E0%A4%93%E0%A4%A1%E0%A4%BF%E0%A4%B6%E0%A4%BE+%E0%A4%A1%E0%A5%87%E0%A4%82%E0%A4%97%E0%A5%82) | HI; disease-specific HTML search | No | Index and a same-host detail route returned HTTP 200 through the bounded collector, and the detail correctly entered `privacy_review_required`. The copyright policy requires prior written NSD/AIR permission, so the route is disabled. Its successful test is a technical seam, not rights-cleared live Hindi coverage. |
| `cbhi_national_health_profile_hi` | [CBHI National Health Profile (Hindi)](https://cbhidghs.mohfw.gov.in/hi/publications/national-health-profile) | HI; official reference | No | Page returned HTTP 200 and first-party material has an attribution-compatible policy, but `/robots.txt` returned 403 and the collector fails closed. It is a manually curated background reference, not a current incident feed or enabled Hindi monitoring route. |
| `ganjam_collectorate` | [Ganjam Collectorate Odia route](https://ganjam.odisha.gov.in/or) | EN/OR; HTML index | Yes | The HTTPS `/or` route is registered; the root's cleartext redirect is not used. Official-source review remains required before full-text redistribution. One district route is not statewide district-source coverage. |
| `dharitri` | [Dharitri](https://www.dharitri.com/) | OR; HTML | No | Permission or a licensed feed is required. Metadata-only discovery remains policy-pending. |
| `prameya` | [Prameya](https://www.prameya.com/) | OR; HTML | No | Permission or a licensed feed is required. Metadata-only discovery remains policy-pending. |
| `odisha_government_press` | [Odisha Government Press](https://govtpress.odisha.gov.in/en/light/odisha-gazettes) | EN/OR; HTML/PDF | No | TLS-chain failure was observed and no certificate bypass is permitted. Government authorship is not automatic reuse permission. |
| `idsp_weekly_outbreaks` | [IDSP weekly listing](https://idsp.mohfw.gov.in/index4.php?lang=1&level=0&linkid=406&lid=3689) | EN; positive-catalogue PDF | Yes | Live origin was unavailable from 25 checked locations. Connector is live-first with HTTPS-CDX-anchored Wayback `id_` fallback. PDFs remain receipt-only until exact-digest approval. Parsed rows are positive catalogue events, never a complete district-week panel. |

The expanded registry includes State portals, Odisha district administration pages and enabled Odisha-scoped Hindi routes. Availability and parser state are reported per route at `/api/v1/sources`; a configured route is not claimed as successful unless the runtime has a valid collection receipt.

## Persistence and public-release boundary

- Enabled live HTML can create a non-content receipt and a disease/place-centred, heuristically redacted evidence span of at most 2,400 characters. Raw bodies are transient.
- Every live evidence span is masked in public signal and assistant responses. Exact bounded redacted spans and protected detail URLs are available only through token-protected review/operator paths.
- Public signal, layer, map and assistant reads include only `processing_state = active_direct` and exclude current `rejected` or `duplicate` decisions. `privacy_review_required`, `language_review_required`, `ambiguous_entity_linkage` and other holds never reach those outputs.
- A PDF digest allowlist authorises one bounded parse attempt for exactly those bytes; it is not a malware, rights, privacy or accuracy verdict.

## Data and geometry inputs

| ID | Source | Licence / policy | Implemented use and caveat |
|---|---|---|---|
| `datameet_census_2011_districts` | [Pinned DataMeet Census 2011 districts](https://github.com/datameet/maps/tree/b3fbbde595310b397a55d718e0958ce249a4fa1f/Districts/Census_2011) | CC BY 2.5 India | Derived Odisha GeoJSON has exactly 30 districts; SHA-256 `8ad5fcc58dffa9d5c99b73a49c65fd3d49e78119cf673e36db99fdaa03fad470`. Community demo geometry, not current operational boundaries. |
| `geoboundaries_ind_adm3` | [geoBoundaries IND ADM3](https://www.geoboundaries.org/api/current/gbOpen/IND/ADM3/) | ODbL 1.0 | Geometry exists, but no authorised tahasil↔block/CHC/PHC/ULB health crosswalk or matching surveillance facts are implemented. No tahasil health map is shipped. |
| `epiclim` | [Zenodo 14580510](https://zenodo.org/records/14580510) | Record terms apply | Audit and positive-catalogue use only. Frozen MD5 `a6c961b95a454226e4720ae1745f9f16`; never a routine count target. |
| `nasa_power` | [NASA POWER daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) | NASA acknowledgement/terms | One configured point for seven days per run, with issue-time receipt validation. Coarse context only; not district weather or causal evidence. |
| `ncvbdc_malaria_annual` | [NCVBDC annual district malaria reports](https://ncvbdc.mohfw.gov.in/index1.php?font=Normal&lang=1&level=1&lid=3689&sublinkid=5784) | Government publication; attribution/source URL retained | Fifteen annual Odisha tables (2010–2024), exactly 30 districts per year and 450 validated rows. Hashes and exact report URLs are retained. Annual observations are not current-week counts. |
| `ogd_odisha_hmis_district_monthly` | [Odisha district-level monthly HMIS catalogue](https://www.data.gov.in/catalog/item-wise-monthly-hmis-report-district-level-odisha) | Government Open Data Licence–India terms apply | 96 monthly CSV resources from April 2012 through March 2020, exactly 30 districts per month and 2,880 validated rows. Values are provisional facility test/service records, not deduplicated people. |
| `open_meteo_ecmwf_seasonal` | [Open-Meteo Seasonal Forecast API](https://open-meteo.com/en/docs/seasonal-forecast-api) backed by [ECMWF seasonal forecasts](https://www.ecmwf.int/en/forecasts/documentation-and-support/seasonal) | Open-Meteo API CC BY 4.0; ECMWF attribution/terms apply | 30 representative district points, 51 control/perturbed trajectories and three 30-day lead windows. Approximately 36 km, not bias-corrected and not a district-average forecast. |
| `chirps` | [CHIRPS](https://www.chc.ucsb.edu/data/chirps) | Provider terms apply | Only explicit policy/preliminary/final vintage states are implemented. A final product must not overwrite what was available at issue time. |
| `era5_land` | [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) | Copernicus terms apply | Credential, licence and request states are implemented; a request/submission state is never called retrieval. |

## Explicit exclusions

- `nrhmorissa.gov.in`: NXDOMAIN during verification.
- `sambad.in`: no stable acquisition route through the observed managed challenge.
- GADM boundaries: incompatible redistribution/non-commercial restrictions.
- `geohacker/india`: its repository label does not override its README attribution to GADM-derived data.

## State semantics

- `ready`: this deployment has recorded a successful runtime receipt; not a rights or accuracy clearance.
- `registered_uncontacted`: enabled but no successful receipt exists in this deployment.
- `policy_pending`: disabled or retention/processing permission unresolved.
- `unavailable`: a registered connector cannot currently supply a valid receipt.
- `partial_registered_source_contact`: at least one but not every enabled source has a successful receipt.
- `observed_for_registered_sources`: every enabled source has a successful receipt; this still does not mean complete web or district coverage.

Each receipt records source ID, requested/final URL internally, retrieval time, HTTP/content metadata, SHA-256 and acquisition path. Public output exposes the registered source URL, not a potentially identifying detail URL. A parser failure or source outage changes coverage state; it never changes disease state.
