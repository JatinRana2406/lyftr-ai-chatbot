.PHONY: help env up down logs ps test restart clean

help:  ## List targets
	@grep -E "^[a-z-]+:.*##" $(MAKEFILE_LIST) | sed "s/:.*##/ -/"

env:  ## Create .env from .env.example if missing
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example - put your real LLM_API_KEYS in it.")

up: env  ## Build and start backend (:8000) and frontend (:3000)
	docker compose up -d --build
	@echo "Frontend: http://localhost:3000   Backend: http://localhost:8000/health"

down:  ## Stop the stack (the SQLite volume is kept)
	docker compose down

restart: down up  ## Restart the stack

logs:  ## Follow logs
	docker compose logs -f

ps:  ## Show running services
	docker compose ps

test:  ## Run backend tests in a throwaway container
	docker run --rm -v "$(CURDIR)/backend:/src" -w /src python:3.12-slim \
	  sh -c "pip install -q -r requirements-dev.txt && python -m pytest -q"

clean:  ## Stop and delete the SQLite volume too
	docker compose down -v
