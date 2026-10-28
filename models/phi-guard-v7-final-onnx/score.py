# phi-guard v7 ONNX scorer: one model, four outputs per note. Needs only numpy and onnxruntime.
#   score    P(PHI left in a de-identified note)                flag when >= 0.5 (calibrated)
#   harmful  P(a harmful leak), never above score               flag when >= 0.5 (calibrated)
#   level    how much PHI the note exposes, 0-6 (score_1..score_6 = P(level >= k))
#   raw      P(a raw, not yet de-identified, note contains PHI)  flag when >= 0.5
# usage: python score.py notes.csv [scores.csv]   CSV with a "text" column (an "id" column is kept)
#        python score.py --text "Pt [NAME] seen on [DATE] for chest pain."
import csv, json, sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
NAMES = ["none", "trace", "low", "moderate", "high", "very_high", "saturated"]
HEADS = (("note", 0, 1, 0), ("harm", 1, 2, 0), ("rich", 2, 8, 1), ("raw", 8, 9, 1))  # combiner, its output columns, which half of the window vector it reads


class PhiGuard:
    def __init__(self, folder=HERE, threads=4):
        folder = Path(folder)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(folder / "model.onnx"), so, providers=["CPUExecutionProvider"])
        n = json.loads((folder / "note.json").read_text())
        self.c = {p: tuple(np.array(n[f"{p}_{a}.{b}"]).reshape(e - s, -1) if b == "weight" else np.array(n[f"{p}_{a}.{b}"]) for a in "qu" for b in ("weight", "bias"))
                  for p, s, e, _ in HEADS}
        self.t, self.read, self.stride, self.pad = n["t"], n["read"], n["stride"], n["pad"]
        self.threshold = json.loads((folder / "config.json").read_text())["threshold"]

    def windows(self, text):  # 1,024-byte windows every 768 bytes, the last one flush with the end; ids 0-255, padding 256
        b = (text or " ").encode()
        s = list(range(0, max(len(b) - self.read, 0) + 1, self.stride))
        if s[-1] + self.read < len(b):
            s.append(len(b) - self.read)
        ids = np.full((len(s), self.read), self.pad, dtype=np.int64)
        for i, a in enumerate(s):
            w = b[a : a + self.read].decode("utf-8", "ignore").encode()
            ids[i, : len(w)] = np.frombuffer(w, dtype=np.uint8)
        return ids

    def logits(self, text):  # 9 note logits: standard, harmful (<= standard), P(level >= 1..6) (never rising), raw; float64, never saturate
        z, v = self.sess.run(None, {"input_ids": self.windows(text)})
        z, d, out = z.astype(np.float64), v.shape[1] // 2, []
        for p, s, e, half in HEADS:  # the note combiner of each output over all windows of the note
            wq, bq, wu, bu = self.c[p]
            vv = v[:, half * d : (half + 1) * d].astype(np.float64)
            a = z[:, s:e] / self.t + vv @ wq.T + bq
            m = a.max(0)
            x = np.exp(a - m)
            out.append(self.t * (m + np.log(x.sum(0))) + (((x / x.sum(0)).T @ vv) * wu).sum(1) + bu)
        out = np.concatenate(out)
        out[:2], out[2:8] = np.minimum.accumulate(out[:2]), np.minimum.accumulate(out[2:8])
        return out

    def scores(self, text):  # the 9 probabilities
        return 1 / (1 + np.exp(-self.logits(text)))

    def logit(self, text):  # standard leak logit: rank with it (PR-AUC / ROC), very confident notes all round to 1.0 as probabilities
        return float(self.logits(text)[0])

    def score(self, text):  # P(leak)
        return float(self.scores(text)[0])

    def level(self, text):  # richness level 0-6
        return int((self.scores(text)[2:8] >= 0.5).sum())


if __name__ == "__main__":
    g = PhiGuard()
    if sys.argv[1] == "--text":
        z = g.logits(sys.argv[2])
        p = 1 / (1 + np.exp(-z))
        lv = int((p[2:8] >= 0.5).sum())
        print(f"leak score {p[0]:.6f} (logit {z[0]:.4f}) flag {p[0] >= g.threshold} | harmful {p[1]:.6f} flag {p[1] >= g.threshold} | "
              f"level {lv} {NAMES[lv]} | raw-note PHI {p[8]:.6f}")
        sys.exit()
    csv.field_size_limit(sys.maxsize)
    rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
    out = open(sys.argv[2], "w", newline="") if len(sys.argv) > 2 else sys.stdout
    w = csv.writer(out)
    w.writerow(["id", "logit", "score", "flag", "harmful_score", "harmful_flag", "level", "level_name"] + [f"score_{k}" for k in range(1, 7)] + ["raw_score", "raw_flag"])
    for i, r in enumerate(rows):
        z = g.logits(r["text"])
        p = 1 / (1 + np.exp(-z))
        lv = int((p[2:8] >= 0.5).sum())
        w.writerow([r.get("id", i), f"{z[0]:.6f}", f"{p[0]:.6f}", int(p[0] >= g.threshold), f"{p[1]:.6f}", int(p[1] >= g.threshold), lv, NAMES[lv]]
                   + [f"{x:.6f}" for x in p[2:8]] + [f"{p[8]:.6f}", int(p[8] >= 0.5)])
