"""Loading CLIMATE-FEVER."""

from datasets import load_dataset

LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]

def load_climate_fever():
    """Return the CLIMATE-FEVER dataset (single, 'test split', 1535 claims)."""
    return load_dataset("tdiggelm/climate_fever", split="test")

def get_claims_and_labels(dataset):
    """Return (claims, label_names) as parallell lists."""
    claims = [ex["claim"] for ex in dataset]
    labels = [LABELS[ex["claim_label"]]for ex in dataset]
    return claims, labels
