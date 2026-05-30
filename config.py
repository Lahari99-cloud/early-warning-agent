"""
Configuration for the Attrition Early-Warning Agent.
"""
# Seed for reproducibility
SEED = 42

# Stable model review bands for predicted 90-day attrition probability.
# These are not performance ratings or employment decisions.
RISK_MEDIUM_THRESHOLD = 0.40
RISK_HIGH_THRESHOLD = 0.70

# Artifact compatibility versions
MODEL_VERSION = "v1"
FEATURE_SCHEMA_VERSION = "v1"

# Versioned artifact registry
ARTIFACT_ROOT = "artifacts"
ACTIVE_MODEL_VERSION = MODEL_VERSION
