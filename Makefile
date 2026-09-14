.PHONY: backend frontend test demo

backend:
	PYTHONPATH=backend uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && pytest -q

demo:
	curl -X POST http://localhost:8000/api/demo/load
