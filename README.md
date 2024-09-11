# CapitalGains

# Import CSVs from brokers
![alt text](image.png)

# Consolidate trades into a standard format
![alt text](image-1.png)

# Performance Reports and Trade reconcilliation 
![alt text](image-2.png)

## Running Locally
- `pipenv install -r requirements.txt`
- `streamlit run main.py`

## Running in Docker
- `docker build -t capitalgains .`
- `docker run -p 8501:8501 capitalgains`

## Development: Running in Docker Volume
- `docker build -t capitalgains .`
- `docker run -v $(pwd)/app:/app -p 8501:8501 capitalgains`
