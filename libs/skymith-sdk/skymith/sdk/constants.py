import os
import warnings

SKYMITH_API_KEY = os.getenv("SKYMITH_API_KEY")
if SKYMITH_API_KEY is None:
    warnings.warn("Skymith API key missing, some endpoints won't work.")
