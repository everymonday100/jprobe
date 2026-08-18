# -*- coding: utf-8 -*-
import os, json, hashlib, argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

GGUF_DIR  = r"E:\OllamaModels\DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
GGUF_FILE = "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
HF_REPO   = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
SF_DIR = r"E:\OllamaModels\Qwen2.5-3B-Instruct\models--Qwen--Qwen2.5-3B-Instruct\snapshots\aa8e72537993ba99e69dfaafa59ed015b17504d1"
STORE_DIR = r"E:\jspace\store"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

class MLPBridge(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.GELU(),
            torch.nn.Linear(dim, dim, bias=False))
    def forward(self, h): return self.net(h)

class SparseBridge(torch.nn.Module):
    """SSAE-style: разреженный инкрементальный код + декодер."""
    def __init__(self, dim, expand=2):
        super().__init__()
        self.enc = torch.nn.Linear(dim, dim * expand)
        self.dec = torch.nn.Linear(dim * expand, dim, bias=False)
    def forward(self, h):
        return self.dec(torch.nn.functional.relu(self.enc(h)))

class MuSparseBridge(torch.nn.Module):
    """SSAE-bottleneck с mu-law компандированием (аналоговый фильтр инкремента)."""
    def __init__(self, dim, mu=255.0, expand=2):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu))
        self.register_buffer("scale", torch.tensor(1.0))
        self.enc = torch.nn.Linear(dim, dim * expand)
        self.dec = torch.nn.Linear(dim * expand, dim, bias=True)   # bias против коллапса в 0
        self.act = torch.nn.GELU()                                 # плавный затвор, не ReLU
    def compand(self, x):
        xn = x / self.scale
        return torch.sign(xn) * torch.log1p(self.mu * torch.abs(xn)) / torch.log1p(self.mu)
    def expand(self, y):
        y = y.clamp(-1, 1)
        return torch.sign(y) * ((1 + self.mu) ** torch.abs(y) - 1) / self.mu * self.scale
    def forward(self, h):
        return self.expand(self.dec(self.act(self.enc(self.compand(h)))))

class DeepBridge(torch.nn.Module):
    """Ёмкий мост (LN + 4 слоя) для богатых трейсов здоровой базы."""
    def __init__(self, dim, hidden=None, layers=4):
        super().__init__()
        hidden = hidden or dim
        mods = [torch.nn.LayerNorm(dim), torch.nn.Linear(dim, hidden), torch.nn.GELU()]
        for _ in range(layers - 2):
            mods += [torch.nn.Linear(hidden, hidden), torch.nn.GELU()]
        mods += [torch.nn.Linear(hidden, dim, bias=False)]
        self.deep = torch.nn.Sequential(*mods)
    def forward(self, h): return self.deep(h)

class JProjector(torch.nn.Module):
    def __init__(self, dim, mode="rms", path=None):
        super().__init__(); self.mode = mode
        if mode == "ln":
            self.norm = torch.nn.LayerNorm(dim).to(device=DEVICE, dtype=DTYPE)
        if mode == "linear":
            self.lin = torch.nn.Linear(dim, dim, bias=False)
            self.lin.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self.lin = self.lin.to(device=DEVICE); self.lin.eval()
        if mode == "mlp":
            sd = torch.load(path, map_location="cpu", weights_only=True)
            if "scale" in sd:                      self.mlp = MuSparseBridge(dim)
            elif any(k.startswith("enc.")  for k in sd): self.mlp = SparseBridge(dim)
            elif any(k.startswith("deep.") for k in sd): self.mlp = DeepBridge(dim)
            else:                                  self.mlp = MLPBridge(dim)
            self.mlp.load_state_dict(sd)
            self.mlp = self.mlp.to(device=DEVICE); self.mlp.eval()
    def forward(self, h):
        if self.mode == "identity": return h.to(DTYPE)
        if self.mode == "ln":       return self.norm(h.to(DTYPE))
        if self.mode == "linear":   return self.lin(h.float()).to(DTYPE)
        if self.mode == "mlp":      return self.mlp(h.float()).to(DTYPE)
        eps = 1e-6
        return (h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps)).to(DTYPE)

class JSpaceStore:
    def __init__(self, root=STORE_DIR):
        self.root = root; os.makedirs(root, exist_ok=True)
    def put(self, tensor, meta):
        arr = tensor.cpu().float().contiguous().numpy()
        data = arr.tobytes()
        key = "idx_" + hashlib.sha256(data).hexdigest()[:16]
        with open(os.path.join(self.root, key + ".bin"), "wb") as f: f.write(data)
        meta.update({"shape": list(arr.shape), "dtype": "float32"})
        with open(os.path.join(self.root, key + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return key
    def get(self, key):
        meta = json.load(open(os.path.join(self.root, key + ".json"), encoding="utf-8"))
        arr = np.fromfile(os.path.join(self.root, key + ".bin"), dtype=np.float32)
        return torch.from_numpy(arr.reshape(meta["shape"]).copy())

def load():
    if os.path.isdir(SF_DIR):
        model = AutoModelForCausalLM.from_pretrained(SF_DIR, dtype=DTYPE).to(DEVICE)
        tok = AutoTokenizer.from_pretrained(SF_DIR)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            GGUF_DIR, gguf_file=GGUF_FILE, dtype=DTYPE).to(DEVICE)
        try:    tok = AutoTokenizer.from_pretrained(GGUF_DIR, gguf_file=GGUF_FILE)
        except Exception: tok = AutoTokenizer.from_pretrained(HF_REPO)
    model.eval()
    return model, tok

def _j_loop(model, tok, prompt, proj, max_steps, max_new, latent_only=False,
            steer_u=None, steer_a=1.0, gate=None):
    base, head, embed = model.model, model.lm_head, model.get_input_embeddings()
    with torch.no_grad():
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt")
        prompt_emb = embed(ids["input_ids"].to(DEVICE))
        emb_max = 3.0 * prompt_emb.norm(dim=-1).mean().item()
        suffix, trace, shadow, diag = [], [], [], []
        h = None
        for t in range(max_steps):
            emb = prompt_emb if h is None else \
                  torch.cat([prompt_emb, torch.cat(suffix, 1)], 1)
            out = base(inputs_embeds=emb.to(DTYPE))
            h_new = out.last_hidden_state[:, -1:, :].detach().float()
            if steer_u is not None:                              # адаптивная проекция
                a_eff = steer_a
                if gate is not None:
                    hn = (h_new.squeeze() - gate["mu"]) / gate["sd"]
                    pabs = torch.softmax(hn @ gate["W_abs"], -1)[1].item()
                    a_eff = steer_a if pabs > gate["thr"] else 0.0
                if a_eff > 0:
                    comp = h_new.squeeze() @ steer_u
                    h_new = h_new - a_eff * comp * steer_u.view(1, 1, -1)
            logits = head(h_new.to(DTYPE))
            if not torch.isfinite(h_new).all(): break
            nt = int(logits.argmax(-1).item())
            ent = -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1).item()
            shadow.append(nt)
            if h is not None:
                cos = F.cosine_similarity(h_new, h, dim=-1).item()
                rel = ((h_new - h).norm() / max(h.norm().item(), 1e-6)).item()
                diag.append((t, round(1 - cos, 3), round(rel, 3), round(ent, 3)))
            h = h_new
            trace.append(h.squeeze(0).squeeze(0).cpu())
            if nt == tok.eos_token_id:
                suffix.append(embed(torch.tensor([[nt]], device=DEVICE)).detach()); break
            if len(shadow) >= 4 and len(set(shadow[-4:])) == 1: break
            e = proj(h)
            n = e.norm(dim=-1, keepdim=True)
            e = e * (torch.clamp(n, max=emb_max) / n.clamp(min=1e-9))
            suffix.append(e)
        shadow_txt = tok.decode(shadow, skip_special_tokens=False)
        gens = []
        for _ in range(max_new):
            emb = torch.cat(suffix, 1) if latent_only else \
                  torch.cat([prompt_emb, torch.cat(suffix, 1)], 1)
            out = base(inputs_embeds=emb.to(DTYPE))
            h_new = out.last_hidden_state[:, -1:, :].detach()
            nt = int(head(h_new.to(DTYPE)).argmax(-1).item())
            if nt == tok.eos_token_id: break
            gens.append(nt)
            suffix.append(embed(torch.tensor([[nt]], device=DEVICE)).detach())
        T = torch.stack(trace)
    return T, diag, shadow_txt, tok.decode(gens, skip_special_tokens=True)

def decode(model, tok, T):
    with torch.no_grad():
        toks = model.lm_head(T.to(DTYPE).to(DEVICE)).argmax(-1)
    return tok.decode(toks.tolist(), skip_special_tokens=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["think", "answer", "latent", "decode",
                                    "baseline", "convert", "shell"])
    ap.add_argument("arg", nargs="?", default="")
    ap.add_argument("--proj", default="rms", choices=["identity", "rms", "ln", "linear", "mlp"])
    ap.add_argument("--proj-path", default=r"E:\jspace\proj.pt")
    ap.add_argument("--max-steps", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--steer", action="store_true")
    a = ap.parse_args()
    model, tok = load()
    store = JSpaceStore()

    steer_cfg = None
    if a.steer:
        d = torch.load(r"E:\jspace\steer.pt", weights_only=True)
        steer_cfg = dict(u=d["u"].to(DEVICE), mu=d["mu"].to(DEVICE),
                         sd=d["sd"].to(DEVICE), W_abs=d["W_abs"].to(DEVICE), thr=0.20)

    def run_think(prompt, want_answer, latent_only=False):
        proj = JProjector(model.config.hidden_size, a.proj, a.proj_path)
        T, diag, shadow, ans = _j_loop(
            model, tok, prompt, proj, a.max_steps, a.max_new if want_answer else 0,
            latent_only,
            steer_u=steer_cfg["u"] if steer_cfg else None,
            steer_a=0.5, gate=steer_cfg)
        key = store.put(T, {"prompt": prompt, "projector": a.proj,
                            "steps": T.shape[0], "shadow": shadow})
        print("ключ:", key, "| шагов:", T.shape[0])
        print("хвост динамики (t, 1-cos, relL2, ent):", diag[-3:])
        print("ТЕНЕВОЙ ТЕКСТ ЦИКЛА:", shadow)
        if want_answer:
            print("ОТВЕТ ПОСЛЕ ЛАТЕНТНОГО ЦИКЛА:", ans)
        return key

    def run_decode(key): print(decode(model, tok, store.get(key)))

    def run_baseline(prompt):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt")
        ids = ids["input_ids"].to(DEVICE)
        with torch.no_grad(): out = model.generate(ids, max_new_tokens=512)
        print(tok.decode(out[0][ids.shape[1]:].tolist(), skip_special_tokens=True))

    if a.cmd == "convert":
        model.save_pretrained(SF_DIR); tok.save_pretrained(SF_DIR)
        print("сохранено в", SF_DIR)
    elif a.cmd == "think":    run_think(a.arg, False)
    elif a.cmd == "answer":   run_think(a.arg, True)
    elif a.cmd == "latent":   run_think(a.arg, True, True)
    elif a.cmd == "decode":   run_decode(a.arg)
    elif a.cmd == "baseline": run_baseline(a.arg)
    elif a.cmd == "shell":
        proj_mode = a.proj
        print("shell: proj <режим> | think/answer/latent/baseline <промт> | decode <ключ> | exit")
        while True:
            try: line = input("j> ").strip()
            except EOFError: break
            if not line: continue
            c, _, arg = line.partition(" ")
            arg = arg.strip()
            if c in ("exit", "quit"): break
            elif c == "proj": proj_mode = arg; print("projector:", proj_mode)
            elif c in ("think", "answer", "latent", "baseline") and arg:
                old = a.proj; a.proj = proj_mode
                if c == "baseline": run_baseline(arg)
                else: run_think(arg, c in ("answer", "latent"), c == "latent")
                a.proj = old
            elif c == "decode" and arg: run_decode(arg)
            else: print("?")