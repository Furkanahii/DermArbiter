# ╔══════════════════════════════════════════════════════════════════╗
# ║  DermArbiter — Colab T4 Tool Validation Notebook               ║
# ║  Her hücreyi sırasıyla Colab'a kopyalayıp çalıştırın.          ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Gereksinimler:
#   - Colab Runtime: T4 GPU (Runtime → Change runtime type → T4)
#   - DermArbiter.zip dosyası Desktop'ta hazır
#   - HuggingFace token (DermoGPT ve MedGemma için)

# ═══════════════════════════════════════════════════════════════════
# CELL 1: GPU Kontrol
# ═══════════════════════════════════════════════════════════════════

# Bu hücreyi çalıştırarak GPU'nun T4 olduğunu doğrulayın
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"VRAM:    {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════
# CELL 2: Zip'i Yükle ve Aç
# ═══════════════════════════════════════════════════════════════════

# Sol panelden Files'a tıklayıp DermArbiter.zip'i yükleyin,
# veya bu hücreyi çalıştırın (otomatik dosya seçici açılır):
from google.colab import files
uploaded = files.upload()  # DermArbiter.zip seçin

# Zip'i aç
!mkdir -p /content/DermArbiter && cd /content/DermArbiter && unzip -o /content/DermArbiter.zip -d . > /dev/null
%cd /content/DermArbiter
!ls -la


# ═══════════════════════════════════════════════════════════════════
# CELL 3: Bağımlılıkları Kur
# ═══════════════════════════════════════════════════════════════════

%%time
# Temel bağımlılıklar
!pip install -e ".[dev]" -q

# Colab-specific ekstra paketler
!pip install bitsandbytes>=0.43.0 accelerate>=0.30.0 -q

# Doğrulama
import timm, open_clip, transformers, chromadb, networkx
print(f"✅ timm:         {timm.__version__}")
print(f"✅ open_clip:    {open_clip.__version__}")
print(f"✅ transformers: {transformers.__version__}")
print(f"✅ chromadb:     {chromadb.__version__}")
print(f"✅ networkx:     {networkx.__version__}")


# ═══════════════════════════════════════════════════════════════════
# CELL 4: HuggingFace Token Ayarla
# ═══════════════════════════════════════════════════════════════════

import os

# Yöntem 1: Colab Secrets (önerilen)
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("✅ HF_TOKEN set from Colab Secrets")
except Exception:
    pass

# Yöntem 2: Manuel giriş (eğer Secrets yoksa)
if not os.environ.get("HF_TOKEN"):
    import getpass
    token = getpass.getpass("HuggingFace Token (boş bırakabilirsiniz): ")
    if token:
        os.environ["HF_TOKEN"] = token
        print("✅ HF_TOKEN set manually")
    else:
        print("⚠️  HF_TOKEN boş — DermoGPT ve MedGemma atlanacak")

# Login
if os.environ.get("HF_TOKEN"):
    !huggingface-cli login --token $HF_TOKEN 2>/dev/null || true


# ═══════════════════════════════════════════════════════════════════
# CELL 5: Gerekli Dizinleri Oluştur
# ═══════════════════════════════════════════════════════════════════

!mkdir -p weights data results data/chroma_cases data/chroma_guidelines
print("✅ Dizin yapısı hazır")


# ═══════════════════════════════════════════════════════════════════
# CELL 6: Tool Validation Çalıştır 🚀
# ═══════════════════════════════════════════════════════════════════

!python scripts/validate_tools_colab.py


# ═══════════════════════════════════════════════════════════════════
# CELL 7 (Opsiyonel): Mock Benchmark Çalıştır
# ═══════════════════════════════════════════════════════════════════

# Mock pipeline ile evaluation modülünü test et
!python -m dermarbiter.evaluation.benchmark_runner --mock --max-cases 10 --output results/


# ═══════════════════════════════════════════════════════════════════
# CELL 8 (Opsiyonel): Sonuçları İncele
# ═══════════════════════════════════════════════════════════════════

import json
from pathlib import Path

# Tool validation sonuçları
val_path = Path("results/tool_validation.json")
if val_path.exists():
    with open(val_path) as f:
        results = json.load(f)
    print("Tool Validation Results:")
    print("-" * 60)
    for r in results:
        icon = "✅" if r["status"] == "PASS" else "❌" if r["status"] in ("FAIL","ERROR") else "⏭️"
        print(f"  {icon} {r['tool_name']:<25s} {r['status']:<8s} {r['elapsed_ms']:>8.0f}ms")
        if r["notes"]:
            print(f"     └─ {r['notes']}")
else:
    print("⚠️ results/tool_validation.json not found — Cell 6'yı çalıştırın")


# ═══════════════════════════════════════════════════════════════════
# CELL 9 (Opsiyonel): VRAM Profili
# ═══════════════════════════════════════════════════════════════════

import torch
if torch.cuda.is_available():
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"Max alloc: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    # VRAM kullanım geçmişi
    torch.cuda.reset_peak_memory_stats()
    print("✅ Peak memory stats reset")
