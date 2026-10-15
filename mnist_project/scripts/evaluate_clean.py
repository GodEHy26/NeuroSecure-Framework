"""Re-evaluate the saved models on the clean MNIST test set.

Recomputes the Accuracy and Test_Loss columns of results/training_results.csv from the
.pth files (the training durations cannot be recomputed, so they are left untouched).
By default it only reports mismatches; pass --write to update the CSV in place.
"""
import argparse
import csv

from common import TRAINING_CSV, evaluate, get_device, load_model, test_tensors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true", help="write the recomputed values back to the CSV")
    args = p.parse_args()

    device = get_device()
    x, y = test_tensors(device)
    with open(TRAINING_CSV) as f:
        rows = list(csv.DictReader(f))

    mismatches = 0
    for r in rows:
        acc, loss, _ = evaluate(load_model(r["Model_Type"], int(r["Epochs"]), int(r["Trial"]), device), x, y)
        acc, loss = round(acc, 2), round(loss, 4)
        if r.get("Accuracy") and (abs(float(r["Accuracy"]) - acc) > 0.01 or abs(float(r["Test_Loss"]) - loss) > 1e-3):
            mismatches += 1
            print(f"MISMATCH {r['Model_Type']} ep{r['Epochs']} t{r['Trial']}: "
                  f"csv {r['Accuracy']}/{r['Test_Loss']} vs recomputed {acc}/{loss}")
        r["Accuracy"], r["Test_Loss"] = acc, loss
        print(f"{r['Model_Type']} ep{int(r['Epochs']):02d} t{r['Trial']}: acc={acc} loss={loss}")

    print(f"\n{len(rows)} models evaluated, {mismatches} mismatches with {TRAINING_CSV}")
    if args.write:
        with open(TRAINING_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Model_Type", "Epochs", "Trial", "Duration_Seconds", "Accuracy", "Test_Loss"])
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
