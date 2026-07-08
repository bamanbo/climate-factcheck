"""Run the majority-class baseline on CLIMATE-FEVER and print scores."""

from factcheck.baseline import majority_class, predict_majority
from factcheck.data import get_claims_and_labels, load_climate_fever
from factcheck.evaluate import full_report, macro_f1

def main():
    dataset = load_climate_fever()
    claims, labels = get_claims_and_labels(dataset)

    majority = majority_class(labels)
    print(f"Majority class: {majority}")

    preds = predict_majority(claims, majority)
    print(f"\nMacro-F1: {macro_f1(labels, preds):.4f}\n")
    print(full_report(labels, preds))

if __name__ == "__main__":
    main()
