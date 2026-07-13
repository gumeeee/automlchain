# 🏃 Teste: LocalProvider

Este notebook testa o **LocalProvider** do AutoMLChain - fine-tuning local usando GPU.

## Requisitos

### Hardware
- **GPU NVIDIA** com CUDA (recomendado) OU **Mac Silicon** (Apple M1/M2)
- **~5GB** de espaço em disco para modelo + dependências
- **~8GB** de RAM (16GB recomendado)

### Software
- Python 3.10+
- PyTorch
- transformers
- peft
- accelerate
- datasets
- bitsandbytes (para QLoRA)

## Tempo Estimado

| Etapa | Tempo |
|-------|-------|
| Instalação deps | ~10 min |
| Download modelo | ~5 min |
| Fine-tuning (1 epoch, 10 samples) | ~15-30 min (GPU) |

## Como Executar

### No Google Colab (Recomendado)

1. Abra o notebook no Colab
2. Runtime > Change runtime type > **GPU** (T4 ou superior)
3. Execute as células em ordem

### Localmente

```bash
# Instalar dependências
pip install torch transformers peft accelerate datasets bitsandbytes

#克隆 repo
git clone https://github.com/gumeeee/automlchain.git
cd automlchain

# Executar notebook
jupyter notebook notebooks/test_local_provider.ipynb
```

## Estrutura do Notebook

1. **Instalar Dependências** - Instala PyTorch e transformers
2. **Preparar Dataset** - Cria dataset de exemplo
3. **Testar LocalProvider** - Inicializa o provider
4. **Iniciar Fine-tuning** - Treina o modelo
5. **Monitorar Progresso** - Acompanha o treinamento
6. **Testar Inferência** - Testa o modelo fine-tuned

## Modelos Recomendados

| Modelo | Tamanho | VRAM | Uso |
|--------|---------|------|-----|
| TinyLlama/TinyLlama-1.1B | 1GB | 2GB | Testes rápidos |
| distilbert/distilgpt2 | 350MB | 1GB | CPU aceitável |
| Qwen/Qwen2-0.5B | 1GB | 2GB | Bom custo-benefício |
| meta-llama/Llama-3.2-1B | 1GB | 2GB | Mais capaz |

## Troubleshooting

### "CUDA out of memory"
Reduza o batch_size ou use QLoRA:
```python
hyperparameters={
    "batch_size": 1,  # Reduzir
    "use_qlora": True,  # Ativar QLoRA
}
```

### "Module not found"
Reinstale as dependências:
```bash
pip install --upgrade torch transformers peft accelerate
```

### "No GPU found"
O fine-tuning em CPU é muito lento (~10x). Recomendamos usar GPU.
