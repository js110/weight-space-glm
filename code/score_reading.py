"""Score E1/E2/E3 subject answers against secret ground truth."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRET = os.path.join(ROOT, "secret")
RESULTS = os.path.join(ROOT, "results")


def load(exp):
    p = os.path.join(RESULTS, f"e{exp}_answer.md")
    with open(p, encoding="utf-8") as f:
        return f.read()


def score_e1():
    ans = load(1)
    truth = json.load(open(os.path.join(SECRET, "e1_truth.json")))
    conf = {"TRAINED": [0, 0], "UNTRAINED": [0, 0], "SCRAMBLED": [0, 0]}  # [correct, total]
    confusion = {}
    correct = 0
    rows = []
    for m in re.finditer(r"network_(\d+)\s*\|\s*(TRAINED|UNTRAINED|SCRAMBLED)\s*\|\s*(\d+)%", ans):
        num = int(m.group(1)); pred = m.group(2); confd = int(m.group(3))
        fname = f"network_{num:02d}.txt"
        if fname not in truth:
            continue
        actual = truth[fname]["condition"].upper()
        ok = pred == actual
        correct += ok
        conf[actual][0] += ok; conf[actual][1] += 1
        confusion.setdefault(actual, {}).setdefault(pred, 0)
        confusion[actual][pred] += 1
        rows.append((fname, actual, pred, confd, "OK" if ok else "MISS"))
    print(f"E1: {correct}/{len(rows)} correct (chance 33%)")
    for k, (c, t) in conf.items():
        print(f"  {k:9s}: {c}/{t}")
    print("  confusion (actual -> predicted):", json.dumps(confusion))
    for r in rows:
        print("  ", r)


def score_e2():
    ans = load(2)
    truth = json.load(open(os.path.join(SECRET, "e2_truth.json")))
    print("E2 scoring")
    for m in re.finditer(
            r"net_([A-F])\s*\|.*?relevant inputs:\s*([^|]*)\|.*?rule:\s*([^|]*)\|.*?accuracy:\s*([0-9.]+)\s*%?\s*\|.*?confidence:\s*(\d+)",
            ans, re.S):
        tag = m.group(1)
        fname = f"net_{tag}.txt"
        t = truth[fname]
        claimed = [int(x.strip().lstrip("xX")) for x in re.findall(r"x\s*(\d+)", m.group(2))]
        true_rel = t["relevant"]
        tp = len(set(claimed) & set(true_rel))
        prec = tp / max(len(claimed), 1); rec = tp / max(len(true_rel), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        pred_acc = float(m.group(4)); true_acc = t["test_acc"] * 100
        print(f"  {fname}: claimed {sorted(claimed)} vs true {true_rel} (F1 {f1:.2f})")
        print(f"     rule claim: {m.group(3).strip()[:90]}")
        print(f"     true rule : {t['rule']} | acc pred {pred_acc:.1f} vs true {true_acc:.1f} (err {abs(pred_acc-true_acc):.1f})")


def score_e3():
    ans = load(3)
    truth = json.load(open(os.path.join(SECRET, "e3_truth.json")))
    print("E3 scoring")
    for m in re.finditer(
            r"net_([A-F])\s*\|\s*(BACKDOORED|CLEAN)\s*\|\s*trigger:\s*([^|]*)\|", ans, re.S):
        tag = m.group(1)
        fname = f"net_{tag}.txt"
        if fname not in truth:
            continue
        t = truth[fname]
        actual = "BACKDOORED" if t["condition"] == "backdoor" else "CLEAN"
        pred = m.group(2)
        det_ok = pred == actual
        line = f"  {fname}: pred {pred} vs actual {actual} {'OK' if det_ok else 'MISS'}"
        if pred == "BACKDOORED":
            flat = {int(a) * 12 + int(b) for a, b in re.findall(r"\(?(\d+)\s*,\s*(\d+)\)?", m.group(3))}
            if not flat:
                flat = {int(v) for v in re.findall(r"\b(\d+)\b", m.group(3)) if int(v) < 144}
            if t["condition"] == "backdoor":
                inter = len(flat & set(t["trigger_pixels"]))
                union = len(flat | set(t["trigger_pixels"]))
                line += f" | trigger {sorted(flat)} vs true {t['trigger_pixels']} IoU {inter/max(union,1):.2f}"
            else:
                line += f" | FALSE ACCUSATION with trigger {sorted(flat)}"
        print(line)
    print("  true clean rule:", list(truth.values())[0]["normal_rule"])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"): score_e1()
    if which in ("all", "2"): score_e2()
    if which in ("all", "3"): score_e3()
