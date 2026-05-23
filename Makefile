install:
	pip install -r requirements.txt

train:
	python3 app/train_models.py

run:
	uvicorn app.main:app --reload

docs:
	open http://127.0.0.1:8000/docs

freeze:
	pip freeze > requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

git-status:
	git status