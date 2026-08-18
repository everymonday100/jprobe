# -*- coding: utf-8 -*-
import json, random, torch
from jspace import load, DEVICE
from finetune_math import (inject_lora, eval_acc, last_number,
                           LORA_PATH, DATA, SEED)
from probe5 import TEST
from probe import prompt_of, result_of

def apply_lora(model):
    params = inject_lora(model)                      # B=0 => пока нет эффекта
    saved = torch.load(LORA_PATH, map_location="cpu", weights_only=True)
    assert len(saved) == len(params), "несовпадение числа LoRA-тензоров"
    with torch.no_grad():
        for p, s in zip(params, saved):
            p.copy_(s.to(DEVICE))
    return len(params) // 2

def acc_test(model, tok, max_new=160):
    ok = 0
    for a, o, b in TEST:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt_of(a, o, b)}],
                                      add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        got = last_number(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
        ok += (got == result_of(a, o, b))
    return ok / len(TEST)

def main():
    random.seed(SEED)
    data = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    random.shuffle(data)
    ev = data[:40]

    model, tok = load()
    print("base  eval:", round(eval_acc(model, tok, ev), 3), "| TEST:", round(acc_test(model, tok), 3))
    n = apply_lora(model)
    print(f"LoRA применена ({n} целевых линейных слоёв)")
    print("lora  eval:", round(eval_acc(model, tok, ev), 3), "| TEST:", round(acc_test(model, tok), 3))

if __name__ == "__main__":
    main()