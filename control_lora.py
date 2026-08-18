# -*- coding: utf-8 -*-
import torch
from jspace import load, DEVICE, DTYPE
from finetune_math import inject_lora, last_number, LORA_PATH
from probe import prompt_of, result_of, corr, acc, one_hot, OPS
from probe5 import TEST, BINS

STEER = r"E:\jspace\steer.pt"

def one_hot_device(y, k):
    return torch.eye(k, device=y.device)[y.long()]

def apply_lora(model):
    params = inject_lora(model)
    saved = torch.load(LORA_PATH, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for p, s in zip(params, saved): p.copy_(s.to(DEVICE))

def main():
    model, tok = load()
    apply_lora(model)
    d = torch.load(STEER, weights_only=True)
    mu, sd = d["mu"].to(DEVICE), d["sd"].to(DEVICE)
    W_op, W_A, W_B = d["W_op"].to(DEVICE), d["W_A"].to(DEVICE), d["W_B"].to(DEVICE)

    Xs, Ss, yop, yA, yB, surf_ok = [], [], [], [], [], 0
    for a, o, b in TEST:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}],
                                      add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=160, do_sample=False)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        surf_ok += (last_number(ans) == result_of(a, o, b))

        full = tok.encode(p + "\n" + ans, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            hs = model.model(full, output_hidden_states=True).hidden_states[-1][0]
        n = hs.shape[0]
        Xs.append((hs - mu) / sd); Ss.append(torch.arange(n))
        yop += [OPS.index(o)] * n; yA += [float(a)] * n; yB += [float(b)] * n

    X = torch.cat(Xs); S = torch.cat(Ss)
    yop = torch.tensor(yop).to(DEVICE)
    yA = torch.tensor(yA).to(DEVICE)
    yB = torch.tensor(yB).to(DEVICE)
    print(f"surface acc (TEST): {surf_ok/len(TEST):.2f}")
    print(f"{'bin':>8s} {'op':>5s} {'A':>6s} {'B':>6s}")
    for lo, hi in BINS:
        m = (S >= lo) & (S < hi)
        if m.sum() < 5: continue
        Xb = X[m]
        print(f"{lo:3d}-{hi:3d}  {acc(W_op, Xb, one_hot_device(yop[m],4)):.2f} "
              f"{corr((Xb@W_A).squeeze(), yA[m]):+.2f} {corr((Xb@W_B).squeeze(), yB[m]):+.2f}")

if __name__ == "__main__":
    main()