from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"D:\CBL\CBL_17_London")

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "police_workforce_selected_forces.csv"

# Workforce data from Table 11 and Table 12
# Table 11: Police officers and total paid workforce
# Table 12: Police officers per 100,000 resident population

workforce = pd.DataFrame([
    {
        "force_slug": "city-of-london",
        "police_force_name": "City of London Police",
        "workforce_table_name": "London, City of",
        "police_officers": 999,
        "total_paid_workforce": 1639,
        "officers_per_100k_population": np.nan,
    },
    {
        "force_slug": "metropolitan",
        "police_force_name": "Metropolitan Police Service",
        "workforce_table_name": "Metropolitan Police",
        "police_officers": 32332,
        "total_paid_workforce": 45533,
        "officers_per_100k_population": 356,
    },
    {
        "force_slug": "west-midlands",
        "police_force_name": "West Midlands Police",
        "workforce_table_name": "West Midlands",
        "police_officers": 8027,
        "total_paid_workforce": 12188,
        "officers_per_100k_population": 264,
    },
    {
        "force_slug": "merseyside",
        "police_force_name": "Merseyside Police",
        "workforce_table_name": "Merseyside",
        "police_officers": 4172,
        "total_paid_workforce": 6655,
        "officers_per_100k_population": 283,
    },
    {
        "force_slug": "west-yorkshire",
        "police_force_name": "West Yorkshire Police",
        "workforce_table_name": "West Yorkshire",
        "police_officers": 6130,
        "total_paid_workforce": 10499,
        "officers_per_100k_population": 252,
    },
    {
        "force_slug": "south-yorkshire",
        "police_force_name": "South Yorkshire Police",
        "workforce_table_name": "South Yorkshire",
        "police_officers": 3040,
        "total_paid_workforce": 5312,
        "officers_per_100k_population": 213,
    },
    {
        "force_slug": "northumbria",
        "police_force_name": "Northumbria Police",
        "workforce_table_name": "Northumbria",
        "police_officers": 3837,
        "total_paid_workforce": 5958,
        "officers_per_100k_population": 254,
    },
    {
        "force_slug": "leicestershire",
        "police_force_name": "Leicestershire Police",
        "workforce_table_name": "Leicestershire",
        "police_officers": 2267,
        "total_paid_workforce": 3896,
        "officers_per_100k_population": 193,
    },
    {
        "force_slug": "nottinghamshire",
        "police_force_name": "Nottinghamshire Police",
        "workforce_table_name": "Nottinghamshire",
        "police_officers": 2380,
        "total_paid_workforce": 3990,
        "officers_per_100k_population": 200,
    },
    {
        "force_slug": "humberside",
        "police_force_name": "Humberside Police",
        "workforce_table_name": "Humberside",
        "police_officers": 2284,
        "total_paid_workforce": 3662,
        "officers_per_100k_population": 237,
    },
])

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
workforce.to_csv(OUTPUT_PATH, index=False)

print("Saved workforce file to:")
print(OUTPUT_PATH)
print()
print(workforce)