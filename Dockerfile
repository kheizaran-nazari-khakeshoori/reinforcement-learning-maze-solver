FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY maze_solver ./maze_solver
COPY train.py evaluate.py config.py ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["pytest", "-q"]
