# -*- coding: utf-8 -*-
# temporal_precedence.py — Wang mediator: does latent op-commitment precede verbalization?
import os, torch
from jspace import load, DEVICE, DTYPE, JProjector, _j_loop
from probe import prompt_of, OPS
from probe5 import TEST

PROJ = os.environ.get("PROJ", r"E:\jspace\proj_deep.pt")
STEER = r"E:\jspace\steer.pt"

def op_margin(h, muS, sdS, W_op):
    """Уверенность op-зонда: logits нормированы, margin = top - mean(остальных)."""
    hn = ((h - muS) / sdS).unsqueeze(0).float()
    logits = (hn @ W_op.float()).squeeze()              # [4]
    logits = logits - logits.mean()                      # центрируем
    order = logits.argsort(descending=True)
    return (logits[order[0]] - logits[order[1:]].mean()).item(), OPS[order[0].item()]

def first_op_token(tok, cot_ids, target_op):
    """Первый токен CoT, вербализующий операцию (грубо: символ операции или слово-маркер)."""
    markers = {"*": ["*", "умнож", "умн"], "+": ["+", "слож", "прибав", "плюс"],
               "-": ["-", "выч", "минус"], "/": ["/", "дел", "раздел"]}
    mk = markers.get(target_op, [])
    for i, tid in enumerate(cot_ids):
        t = tok.decode([tid]).lower()
        if any(m in t for m in mk):
            return i
    return len(cot_ids)

def main():
    model, tok = load()
    proj = JProjector(model.config.hidden_size, "mlp", PROJ)
    d = torch.load(STEER, weights_only=True)
    W_op, muS, sdS = d["W_op"].to(DEVICE), d["mu"].to(DEVICE), d["sd"].to(DEVICE)

    print(f"{'prompt':28s} {'op':>3s} {'verb_step':>9s} {'lat_plateau':>11s} {'lead':>5s} | verdict")
    print("-" * 78)
    leads = []
    for a, o, b in TEST:
        p = prompt_of(a, o, b)
        # латентная траектория (bridge)
        T = _j_loop(model, tok, p, proj, 48, 0)[0]
        margins, preds = [], []
        with torch.no_grad():
            for t in range(T.shape[0]):
                m, pr = op_margin(T[t].to(DEVICE), muS, sdS, W_op)
                margins.append(m); preds.append(pr)
        # плато латентной уверенности: первый шаг, где margin >= 0.8 * max и pred верный
        mx = max(margins)
        plateau = next((t for t, m in enumerate(margins) if m >= 0.8 * mx and preds[t] == o), None)
        # момент вербализации (surface CoT)
        ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad(): out = model.generate(ids, max_new_tokens=160, do_sample=False)
        verb = first_op_token(tok, out[0][ids.shape[1]:], o)
        lead = (verb - plateau) if (plateau is not None) else None
        leads.append(lead)
        v = "✅ PRECEDES" if (lead is not None and lead > 0) else ("⚠️ simult" if lead == 0 else "❌ no/lag")
        print(f"{p:28s} {o:>3s} {verb:>9d} {str(plateau):>11s} {str(lead):>5s} | {v}")

    valid = [l for l in leads if l is not None]
    if valid:
        print(f"\nСредний lead латента над surface: {sum(valid)/len(valid):+.1f} шагов "
              f"({sum(1 for l in valid if l>0)}/{len(valid)} случаев предшествования)")

if __name__ == "__main__":
    main()