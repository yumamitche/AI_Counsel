Local TinyLlama Model Usage

This project supports loading TinyLlama from a local directory to avoid runtime downloads.

1) Download model files to a local folder (example path: ml_models/tinyllama).

   - Preferred (Python):
     - pip install huggingface-hub
     - python download_tinyllama_local.py --output ml_models/tinyllama

   - Alternative (git + git-lfs):
     - Install git and git-lfs
     - git lfs install
     - git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 ml_models/tinyllama

2) Point the application to the local model by setting an environment variable.

   Windows (PowerShell):

```powershell
$env:LOCAL_TINYLLAMA_PATH = "C:\path\to\project\ml_models\tinyllama"
python app.py
```

   Or permanently set in System Environment Variables.

3) The loader will detect `LOCAL_TINYLLAMA_PATH` and load the model locally without attempting to download.

Notes:
- Model files are large (~2GB). Ensure you have enough disk space.
- If the model requires authentication on Hugging Face, run `huggingface-cli login` before downloading.
- If you prefer another local path, set `LOCAL_TINYLLAMA_PATH` to the desired directory containing model files.
