# Reproducibility

- Dockerfile: `docker build -t maze . && docker run maze`
- requirements.txt hashes: see requirements.lock
- pyproject.toml requires-python >=3.10,<3.13
- seed: `train.py --seed 0` deterministic

All results mean+-std over 20 eval seeds.
