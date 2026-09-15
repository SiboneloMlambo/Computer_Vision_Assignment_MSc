# Run this project in Google Colab

1. Push this corrected folder to your GitHub repository.
2. Open `notebooks/colab_reproduction.ipynb` in Google Colab.
3. Select **Runtime → Change runtime type → T4 GPU**.
4. Run the notebook cells from top to bottom.
5. In the calibration cell, replace the sample image and near/far
   coordinates with obvious points from one downloaded DA-2K image.
6. Run the two 20-image smoke tests before starting the full evaluation.

The notebook downloads both official code repositories, both checkpoints,
DA-2K, and the DIODE validation set. Allow roughly 5 GB of free Colab disk
space. V1 and V2 checkpoints are loaded strictly with their matching model
implementations; a weight mismatch stops the run.

## Quick code check

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Expected result: `19 passed`.
