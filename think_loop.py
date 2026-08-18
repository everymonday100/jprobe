# -*- coding: utf-8 -*-
# think_loop.py — Проект 2: латентный think-loop + lens + idx + перенос цепочек
import os, re, json, hashlib, argparse, torch
from jspace import load, DEVICE, DTYPE, JProjector, _j_loop
from probe import prompt_of, result_of, one_hot
from probe5 import BIG, TEST

PROJ_PATH = os.environ.get("PROJ", r"E:\jspace\proj_deep.pt")
CHAINS_PATH = r"E:\jspace\chains.json"
SECOND_DIR = r"E:\OllamaModels\Qwen2.5-0.5B-Instruct"

def ridge_fit_device(X, Y, lam=1e-2):
    Y = Y.to(X.device)
    return torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=X.device), X.T @ Y)

def last_number(txt):
    m = re.findall(r"=\s*(-?\d+)", txt)          # последний «= N» и есть ответ
    if m: return int(m[-1])
    m = re.findall(r"(?:Таким образом|Ответ|итого|получаем)[^\d-]*(-?\d+)", txt, re.I)
    if m: return int(m[-1])
    m = re.findall(r"-?\d+", txt)
    return int(m[-1]) if m else None

def gen_answer(model, tok, user_content, max_new=160):
    ids = tok.apply_chat_template([{"role": "user", "content": user_content}],
                                  add_generation_prompt=True, return_tensors="pt")
    input_ids = ids["input_ids"].to(DEVICE)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new, do_sample=False)
    txt = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return last_number(txt), txt

def think(model, tok, proj, prompt, steps=48):
    """Латентный think-loop: шаги в J-space без эмиссии токенов + lens-вербализация."""
    T = _j_loop(model, tok, prompt, proj, steps, 0)[0]          # [steps, d]
    chain_ids, rows = [], []
    with torch.no_grad():
        for t in range(T.shape[0]):
            h = T[t].to(DEVICE).unsqueeze(0)
            top = torch.topk(model.lm_head(h.to(DTYPE)).float(), 5).indices[0].cpu().tolist()
            rows.append((t, [tok.decode([i]).strip() or "_" for i in top]))
            s0 = tok.decode([top[0]]).strip()
            semantic = bool(s0) and (any(c.isdigit() for c in s0) or s0 in "+-*/=")
            if semantic and (not chain_ids or chain_ids[-1] != top[0]):
                chain_ids.append(top[0])
    idx = hashlib.sha256(",".join(map(str, chain_ids)).encode()).hexdigest()[:16]
    return T, rows, chain_ids, idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--second", action="store_true")
    a = ap.parse_args()

    model, tok = load()
    proj = JProjector(model.config.hidden_size, "mlp", PROJ_PATH)

    chains = {}
    print(f"{'Prompt':34s} {'Exp':>5s} {'base':>6s} {'think':>6s} | idx / chain")
    print("-" * 120)
    for n, (aa, o, bb) in enumerate(TEST):
        p = prompt_of(aa, o, bb)
        base, _ = gen_answer(model, tok, p)
        T, rows, ids, idx = think(model, tok, proj, p, a.steps)
        chain_text = tok.decode(ids)
        adv, _ = gen_answer(model, tok, p + "\n[Внутренние мысли модели]: " + chain_text + "\nДай ответ числом.")
        chains[p] = {"idx": idx, "chain": chain_text, "expected": result_of(aa, o, bb)}
        print(f"{p:34s} {result_of(aa,o,bb):5d} {str(base):>6s} {str(adv):>6s} | {idx} / {chain_text[:48]!r}")
        if n == 0:
            print("  --- lens-строки первого промта (шаг: top-5 thought tokens) ---")
            for t, toks in rows[:12]:
                print(f"    step {t:2d}: {toks}")
    with open(CHAINS_PATH, "w", encoding="utf-8") as f:
        json.dump(chains, f, ensure_ascii=False, indent=1)
    print("Цепочки сохранены в", CHAINS_PATH)
    del model, proj
    torch.cuda.empty_cache()

    if a.second:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from probe5 import BIG
            print("\n=== 0.5B: детектор ошибок (Sun et al.) + гейтированный перенос ===")
            m2 = AutoModelForCausalLM.from_pretrained(SECOND_DIR, torch_dtype=DTYPE).to(DEVICE)
            t2 = AutoTokenizer.from_pretrained(SECOND_DIR)

            def hidden_last(m, t, p):
                ids = t.apply_chat_template([{"role": "user", "content": p}],
                                            add_generation_prompt=True, return_tensors="pt")
                input_ids = ids["input_ids"].to(DEVICE)
                with torch.no_grad():
                    out = m.model(input_ids, output_hidden_states=True)
                return out.hidden_states[-1][0, -1, :].float()

            # 1) детектор корректности на hidden states 0.5B
            Xs, ys = [], []
            for aa, o, bb in BIG:
                p = prompt_of(aa, o, bb)
                num, _ = gen_answer(m2, t2, p, max_new=160)
                Xs.append(hidden_last(m2, t2, p))
                ys.append(1 if num == result_of(aa, o, bb) else 0)
            X = torch.stack(Xs); y = torch.tensor(ys).to(DEVICE)
            W_err = ridge_fit_device(X, one_hot(y, 2))
            print(f"детектор обучен: {sum(ys)}/{len(ys)} верных в BIG")

            # 2) гейтированный перенос: цепочка только если детектор флагует ошибку
            print(f"{'Prompt':34s} {'exp':>5s} {'base':>6s} {'gated':>6s} | gate")
            for p, c in chains.items():
                b2, _ = gen_answer(m2, t2, p, max_new=160)
                h = hidden_last(m2, t2, p)
                with torch.no_grad():
                    pc = torch.softmax((h @ W_err).unsqueeze(0), -1)[0, 1].item()
                if pc < 0.5:  # детектор считает base-ответ ошибочным
                    g2, _ = gen_answer(m2, t2,
                        p + "\n[Черновик источника]: числа {" + c["chain"] +
                        "}. Заверши вычисление САМОСТОЯТЕЛЬНО и дай ответ числом.", max_new=160)
                    gate = "CHAIN"
                else:
                    g2 = b2; gate = "base"
                print(f"{p:34s} {c['expected']:5d} {str(b2):>6s} {str(g2):>6s} | {gate}(p={pc:.2f})")
            del m2; torch.cuda.empty_cache()
        except Exception as e:
            print("Вторая модель не загрузилась:", e)

if __name__ == "__main__":
    main()