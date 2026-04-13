# Deep Learning Project

## Setup Instructions

1. **Create and Activate a Virtual Environment:**
   Using `venv`:
   ```bash
   python -m venv venv
   ```
   * Windows: `venv\Scripts\activate`
   * Mac/Linux: `source venv/bin/activate`

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch with CUDA 12.8:**
   If you have the CPU version installed, uninstall it first:
   ```bash
   pip uninstall torch torchvision
   ```
   Then install the CUDA-enabled version:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
