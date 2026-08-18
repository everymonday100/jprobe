# -*- coding: utf-8 -*-
import os, re, json, math, random, torch, torch.nn as nn
from jspace import load, DEVICE, DTYPE

DATA = r"E:\jspace\finetune_data.jsonl"
LORA_PATH = r"E:\jspace\lora_math.pt"
TARGETS = ("q_proj", "v_proj"); R, ALPHA = 8, 16
MAX_LEN, EPOCHS, LR, ACCUM, EVAL_N, SEED = 512, 2, 1e-3, 8, 40, 7

# ---------- детерминированные решения ----------
def sol_of(ex):
    p, a, cat = ex["prompt"], ex["answer"], ex["category"]
    try:
        if cat.startswith("arithmetic"):
            m = re.search(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)", p)
            x, op, y = int(m.group(1)), m.group(2), int(m.group(3))
            return f"{x} {op} {y} = {a}.\nОтвет: {a}."
        if cat == "two_step":
            expr = re.search(r"будет (.+?)\?", p).group(1)
            inner = re.search(r"\(([^)]+)\)", expr).group(1)
            iv = eval(inner, {"__builtins__": {}})
            return f"Скобки: {inner} = {iv}. Затем: {expr} = {a}.\nОтвет: {a}."
        if cat == "linear":
            A = int(re.search(r"(\d+)x", p).group(1))
            B = int(re.search(r"\+ \(?(-?\d+)\)? =", p).group(1))
            C = int(re.search(r"= (-?\d+)", p).group(1))
            return f"{A}x = {C} - ({B}) = {C-B}. x = {C-B}/{A} = {a}.\nОтвет: {a}."
        if cat == "quadratic":
            A = int(re.search(r"(\d+)x²", p).group(1))
            Bm = re.search(r"([+-]\d+)x", p); B = int(Bm.group(1)) if Bm else 0
            Cm = re.search(r"([+-]\d+) =", p); C = int(Cm.group(1)) if Cm else 0
            D = B*B - 4*A*C; s = int(math.isqrt(D))
            return f"D = {B}²-4*{A}*{C} = {D}, √D = {s}. x = ({-B}±{s})/{2*A} = {a}.\nОтвет: {a}."
        nums = re.findall(r"\d+", p)
        return f"Вычислим по условию: {a}.\nОтвет: {a}."
    except Exception:
        return f"Ответ: {a}."

# ---------- hook-based LoRA ----------
class LoRAHook:
    def __init__(self, mod, r=R, alpha=ALPHA):
        self.A = nn.Parameter(torch.randn(r, mod.in_features, device=DEVICE) * 0.01)
        self.B = nn.Parameter(torch.zeros(mod.out_features, r, device=DEVICE))
        self.scale = alpha / r
        mod.register_forward_hook(self.hook)
    def hook(self, mod, args, out):
        x = args[0]
        d = (x.float() @ self.A.T @ self.B.T) * self.scale
        return out + d.to(out.dtype)

def inject_lora(model):
    params = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and any(t in n for t in TARGETS):
            h = LoRAHook(m); params += [h.A, h.B]
    return params

# ---------- данные ----------
def make_sample(tok, p, sol):
    pr = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids[0]
    fu = tok.apply_chat_template([{"role":"user","content":p},{"role":"assistant","content":sol}], add_generation_prompt=False, return_tensors="pt").input_ids[0]
    fu = torch.cat([fu, torch.tensor([tok.eos_token_id])])[:MAX_LEN]
    lb = fu.clone(); lb[:min(len(pr), MAX_LEN)] = -100
    return fu, lb

def last_number(t):
    m = re.findall(r"=\s*(-?\d+)", t) or re.findall(r"-?\d+", t)
    return int(m[-1]) if m else None

def eval_acc(model, tok, items):
    ok = 0
    for ex in items:
        ids = tok.apply_chat_template([{"role":"user","content":ex["prompt"]}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=96, do_sample=False)
        got = last_number(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
        ans = ex["answer"]
        target = float(ans[0]) if isinstance(ans, list) else float(ans)   # int/float/list -> float
        ok += (got is not None and abs(got - target) < 1e-6)
    return ok / len(items)

def main():
    random.seed(SEED)
    data = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    random.shuffle(data)
    ev, tr = data[:EVAL_N], data[EVAL_N:]

    model, tok = load()
    model.config.use_cache = False
    print("eval ДО:", round(eval_acc(model, tok, ev), 3))

    params = inject_lora(model)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(params, lr=LR)
    samp = [make_sample(tok, ex["prompt"], sol_of(ex)) for ex in tr]

    step = 0
    for ep in range(EPOCHS):
        for fu, lb in samp:
            fu, lb = fu.unsqueeze(0).to(DEVICE), lb.unsqueeze(0).to(DEVICE)
            with torch.autocast("cuda", dtype=DTYPE):
                loss = model(input_ids=fu, labels=lb).loss / ACCUM
            loss.backward()
            step += 1
            if step % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
        print(f"epoch {ep} loss {loss.item()*ACCUM:.4f}")

    torch.save([p.detach().cpu() for p in params], LORA_PATH)
    print("eval ПОСЛЕ:", round(eval_acc(model, tok, ev), 3), "| LoRA:", LORA_PATH)

if __name__ == "__main__":
    main()