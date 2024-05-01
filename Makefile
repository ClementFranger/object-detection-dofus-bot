create:
	conda create -n object-detection-dofus-bot python=3.11
precommit:
	pre-commit run --all-files
requirements:
	poetry export -f requirements.txt --output requirements.txt
