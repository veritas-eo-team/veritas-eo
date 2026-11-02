# Data: official reference datasets

This project uses the two official reference datasets for TSYP14 Track 2 (False-Data Injection):

| Dataset | Role in the system |
|---|---|
| NASA POWER Daily API | the protected series (target) and its context: neighbouring grid cells and other variables at the site |
| Copernicus Sentinel-2 L2A | independent satellite evidence: cloud fraction (used by the cloud-consistency check) and NDVI (stored for display) |

Both are free. NASA POWER needs no account; Sentinel-2 needs a free Copernicus Data Space Ecosystem account.

## 1. NASA POWER

**Access.** Daily point API (`https://power.larc.nasa.gov/api/temporal/daily/point`), community `AG`, JSON format, up to 20 parameters per request. Requests are retried on network errors and HTTP 429/5xx (waits of 2, 4, 8 s), with a 1 s delay between calls.

**Variables**

| Variable | Meaning | Unit | Used as |
|---|---|---|---|
| ALLSKY_SFC_SW_DWN | all-sky surface shortwave downward irradiance | kWh/m²/day | **target** (the protected series) |
| CLRSKY_SFC_SW_DWN | clear-sky surface shortwave downward irradiance | kWh/m²/day | context feature; physical ceiling for the clear-sky check |
| T2M | air temperature at 2 m | °C | context feature |
| RH2M | relative humidity at 2 m | % | context feature |
| PRECTOTCORR | corrected precipitation | mm/day | context feature (log(1 + x)) |
| GWETTOP | surface soil wetness | 0–1 | downloaded, **excluded** from the model (reflects past rain, not today's sky) |

**Grid.** Solar parameters come on a 1° × 1° grid (CERES SYN1deg); meteorology on 0.5° × 0.625° (MERRA-2).

**Cells per location.** For each location we download five points:

| Cell id | Position | Variables |
|---|---|---|
| `site` | the location itself | target + all context variables |
| `nb_N`, `nb_E`, `nb_S`, `nb_W` | 1° north, east, south, west (longitude wrapped, latitude clipped) | target only |

The 1° offset puts each neighbour in a different solar grid cell. During alignment, a neighbour is dropped if it is identical to the site series (largest difference < 10⁻⁹) or more than 30% missing; kept neighbours become `nb_0`..`nb_3`.

**Cleaning.** Values ≤ −990 (the −999 fill value) become missing. Nothing is interpolated.

**Period.** 2018-01-01 to 2024-12-31, daily.

**Cache.** `data/raw/power/<location>/<cell_id>.json` (not in the repository). A file is cached only after all requested variables are confirmed present, and it is written atomically.

**Download.**

    python -m src.data.power --all
    python -m src.data.power --location tunis [--force]

## 2. Copernicus Sentinel-2 L2A

**Access.** Sentinel Hub Statistical API of the Copernicus Data Space Ecosystem (`https://sh.dataspace.copernicus.eu/api/v1/statistics`), OAuth2 client credentials read from the environment variables `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET`. Credentials are never logged or stored.

**Request.** Collection `sentinel-2-l2a`; a 3 km × 3 km box centred on each location; 20 m resolution; daily aggregation; requested year by year, 2018–2024.

**Per-acquisition outputs** (from the scene classification layer, SCL)

| Column | Definition |
|---|---|
| `valid_fraction` | share of box pixels with data (SCL not 0 or 1) |
| `cloud_fraction` | share of valid pixels in SCL classes 3, 8, 9, 10 (cloud shadow, medium and high cloud, cirrus) |
| `ndvi_mean` | mean of (B08 − B04)/(B08 + B04) over valid, cloud-free pixels |

**Clear image.** cloud_fraction ≤ 0.3 and valid_fraction ≥ 0.5. Locations have between 57 and 781 clear images over 2018–2024.

**Use in the system.**

- `cloud` (cloud fraction on acquisition days with valid_fraction ≥ 0.5) feeds the cloud-consistency check: a heavy-cloud day (≥ 0.7) on which the target reaches ≥ 80% of clear-sky is a conflict.
- `ndvi` (last clear value, carried forward only, so it stays causal) is stored for display; it is not a model feature.

**Measured satellite/ground mismatch.** On clean train data, 826 of 3,338 heavy-cloud Sentinel-2 days (24.7%) look clear in NASA POWER. This is the difference between a 3 km instant view (around 10:30 local time) and a daily mean over a ~100 km cell. The cloud check therefore needs 6 conflicts within 14 days before it alarms.

**Cache.** `data/raw/s2/<location>.csv` (not in the repository).

**Download** (optional; without credentials the system runs, only the cloud check is inactive):

    $env:CDSE_CLIENT_ID="..."        (PowerShell; bash: export CDSE_CLIENT_ID=...)
    $env:CDSE_CLIENT_SECRET="..."
    python -m src.data.sentinel2 --all

## 3. Locations and split

| Location | Latitude | Longitude | Climate | Split |
|---|---|---|---|---|
| tunis | 36.80 | 10.18 | mediterranean | train |
| tozeur | 33.92 | 8.13 | desert | train |
| seville | 37.39 | −5.98 | mediterranean | train |
| phoenix | 33.45 | −112.07 | desert | train |
| paris | 48.86 | 2.35 | temperate | train |
| ames | 42.03 | −93.62 | farmland | train |
| manaus | −3.12 | −60.02 | tropical forest | train |
| nairobi | −1.29 | 36.82 | tropical highland | train |
| black_forest | 48.00 | 8.20 | forest | train |
| ludhiana | 30.90 | 75.85 | farmland | train |
| kairouan | 35.68 | 10.10 | semi-arid | test |
| alice_springs | −23.70 | 133.88 | desert | test |
| lyon | 45.76 | 4.84 | temperate | test |
| abidjan | 5.36 | −4.01 | tropical | test |
| pergamino | −33.89 | −60.57 | farmland | test |

- **Train** locations, years 2018–2023: context model, threshold tuning, fusion model.
- **Test** locations, 2024: main benchmark. Never seen by training or tuning; the code refuses test frames in both.
- **Unseen year**: the train locations in 2024, as a second benchmark.

## 4. Clean, aligned frames

`data/clean/<location>.parquet` (committed, so the demo, app and benchmark run without downloading). One row per day on a continuous daily index, all float64:

| Column | Content |
|---|---|
| `target` | site ALLSKY_SFC_SW_DWN (NaN if missing) |
| `nb_0`..`nb_3` | target variable at the kept neighbour cells |
| `ctx_<VAR>` | other NASA POWER variables at the site |
| `ndvi` | last clear Sentinel-2 NDVI, carried forward |
| `cloud` | Sentinel-2 cloud fraction on acquisition days, else NaN |
| `days_since_clear` | days since the last clear NDVI |

Build or rebuild:

    python -m src.data.align --all

A frame is rebuilt only if an input file hash or a relevant setting changes.

## 5. Provenance

Every downloaded file and every clean frame is recorded in `data/provenance/log.jsonl`: one signed, hash-chained record per file, with the SHA-256 of the file, the API request (download) or the input hashes (derived), the code version, and an Ed25519 signature.

Reference run: 120 records, signing key id `21ded33c4cec6bf2`, head hash `038ee7c91d72c20521f0b32035a0cf426e98a79486253f8daa9c01d8370ec7af`.

Check on a fresh copy (raw data is not in the repository, so file contents are skipped):

    python -m src.trust.verify --allow-missing --trust 21ded33c4cec6bf2 --expected-head 038ee7c91d72c20521f0b32035a0cf426e98a79486253f8daa9c01d8370ec7af

Re-downloading on another machine creates a new signing key and new records. To reproduce the reported results, use the committed clean frames; there is no need to download.

## 6. Acknowledgements

- The data was obtained from the National Aeronautics and Space Administration (NASA) Langley Research Center (LaRC) Prediction of Worldwide Energy Resource (POWER) Project funded through the NASA Earth Science/Applied Science Program.

- Contains modified Copernicus Sentinel data 2018–2024, processed with the Sentinel Hub Statistical API of the Copernicus Data Space Ecosystem.
