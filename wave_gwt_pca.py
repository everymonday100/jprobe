# -*- coding: utf-8 -*-
# wave_gwt_pca.py — Global Workspace Theory PCA capacity test
import os, torch, numpy as np
from sklearn.decomposition import PCA
from jspace import load, DEVICE, DTYPE
from probe import prompt_of
from probe5 import BIG  # используем BIG для разнообразия задач

def collect_global_traces(model, tok, dataset):
    """
    Собирает активации со ВСЕХ слоев на последнем токене промпта.
    Возвращает словарь: {layer_idx: numpy_array [num_examples, hidden_dim]}
    """
    num_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    num_examples = len(dataset)
    
    layer_data = {l: np.zeros((num_examples, hidden_dim), dtype=np.float32) for l in range(num_layers)}
    
    print(f"Сбор глобальных траекторий для {num_examples} примеров...")
    
    for idx, (a, o, b) in enumerate(dataset):
        prompt = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        
        current_step_states = {}
        
        def make_hook(layer_i):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                current_step_states[layer_i] = h[0, -1, :].detach().cpu().float().numpy()
            return hook
        
        handles = []
        for l in range(num_layers):
            handles.append(model.model.layers[l].register_forward_hook(make_hook(l)))
            
        try:
            with torch.no_grad():
                model(ids)
        finally:
            for h in handles:
                h.remove()
                
        for l in range(num_layers):
            layer_data[l][idx] = current_step_states[l]
            
        if (idx + 1) % 10 == 0:
            print(f"  Обработано: {idx + 1}/{num_examples}")
            
    return layer_data

def analyze_gwt_capacity(layer_data, variance_threshold=0.85):
    """Вычисляет эффективную размерность (PCA) для каждого слоя."""
    num_layers = len(layer_data)
    results = []
    
    print(f"\n=== GWT Capacity Analysis (variance threshold={variance_threshold*100:.0f}%) ===")
    print(f"{'Layer':>6s} {'eff_dim':>8s} {'var_1st':>8s} {'var_cum':>8s} | interpretation")
    print("-" * 65)
    
    for l in range(num_layers):
        X = layer_data[l]
        # Центрируем
        X_centered = X - X.mean(axis=0)
        
        # PCA
        pca = PCA(n_components=min(X.shape[0], X.shape[1]))
        pca.fit(X_centered)
        
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        eff_dim = int(np.searchsorted(cumvar, variance_threshold) + 1)
        var_1st = pca.explained_variance_ratio_[0]
        var_cum = cumvar[min(eff_dim-1, len(cumvar)-1)]
        
        # Интерпретация по GWT
        if eff_dim <= 4:
            interp = "🔵 GWT bottleneck (~4 chunks)"
        elif eff_dim <= 10:
            interp = "🟡 moderate capacity"
        else:
            interp = "🔴 high-dim (no compression)"
        
        results.append((l, eff_dim, var_1st, var_cum))
        print(f"{l:>6d} {eff_dim:>8d} {var_1st:>8.3f} {var_cum:>8.3f} | {interp}")
    
    # Сводка
    dims = [r[1] for r in results]
    min_layer = results[np.argmin(dims)][0]
    max_layer = results[np.argmax(dims)][0]
    print(f"\nМинимальная ёмкость: слой {min_layer} (dim={min(dims)})")
    print(f"Максимальная ёмкость: слой {max_layer} (dim={max(dims)})")
    print(f"Средняя эффективная размерность: {np.mean(dims):.1f}")
    
    # GWT проверка: есть ли слой с dim ≈ 4?
    gwt_layers = [r[0] for r in results if r[1] <= 4]
    if gwt_layers:
        print(f"\n✅ GWT bottleneck найден в слоях: {gwt_layers}")
        print("   Это согласуется с гипотезой ~4 chunks рабочей памяти.")
    else:
        print(f"\n❌ GWT bottleneck (dim≤4) не найден.")
        print("   Модель не сжимает информацию до ~4 измерений ни в одном слое.")
    
    return results

def main():
    model, tok = load()
    
    # Используем BIG для разнообразия (32 примера)
    dataset = BIG
    print(f"Датасет: {len(dataset)} примеров из BIG")
    
    layer_data = collect_global_traces(model, tok, dataset)
    results = analyze_gwt_capacity(layer_data, variance_threshold=0.85)
    
    # Дополнительно: анализ на разных порогах
    print(f"\n=== Чувствительность к порогу дисперсии ===")
    for thresh in [0.70, 0.80, 0.90, 0.95]:
        dims = []
        for l in range(len(layer_data)):
            X = layer_data[l] - layer_data[l].mean(axis=0)
            pca = PCA(n_components=min(X.shape[0], X.shape[1]))
            pca.fit(X)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            dims.append(int(np.searchsorted(cumvar, thresh) + 1))
        print(f"  threshold={thresh:.2f}: mean_dim={np.mean(dims):.1f}, min={min(dims)}, max={max(dims)}")

if __name__ == "__main__":
    main()