# STEP UP Impact Dashboard

Dash app for the cleaned STEP UP dashboard data.

## Run

From the project root:

```powershell
python .\dashboard\app.py
```

Then open:

```text
http://127.0.0.1:8050/
```

## Pages

- Overview
- STAR Growth
- STEP UP vs Non-STEP UP
- Attendance and CAASPP
- Student Drilldown
- Data Coverage

## Main Filters

- Student / ID search
- Analysis mode: all available data or strict comparable data
- Student group: all, STEP UP, or Non-STEP UP
- Subject
- STAR year
- STAR period
- School focus
- Grade
- Ethnicity
- Intervention intensity

## Data Source

The app reads the generated CSV files in `clean_data/`. Rebuild them with:

```powershell
python .\scripts\build_dashboard_data.py
```

Strict comparison mode currently stays on the 2022-25 comparison years. 2025-26 STAR and SIS attendance are available in the dashboard, but 2025-26 CAASPP is intentionally left blank until that source exists.
