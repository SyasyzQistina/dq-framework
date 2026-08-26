import pandas as pd
from src.profiling.dimensions import validity

test = pd.Series(["-1.548", "-1.55", "-1.52"])
print(validity(test, "numeric"))

cleaned = test.astype(str).str.replace(",", "").str.strip()
print(pd.to_numeric(cleaned, errors="coerce").tolist())