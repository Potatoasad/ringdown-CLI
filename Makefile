SHELL = /bin/bash
ENV_FOLDER = ./environments
ROOT_FOLDER = .
NOTEBOOK_SCRIPT = jupyter
EXECUTABLE_SCRIPT = python

.ONESHELL:

${ENV_FOLDER}/notebook-venv:
	[[ -d ${ENV_FOLDER} ]] || mkdir ${ENV_FOLDER}
	#venv_folder = ${ENV_FOLDER}/notebook-venv
	#echo venv_folder
	python -m venv ${ENV_FOLDER}/notebook-venv
	source ${ENV_FOLDER}/notebook-venv/bin/activate; pip install --upgrade pip; pip install --upgrade setuptools; pip install -r requirements.txt
	source ${ENV_FOLDER}/notebook-venv/bin/activate; pip install -e ${ROOT_FOLDER}

jupyter-bash-script: ${ENV_FOLDER}/notebook-venv
	rm -f ./${NOTEBOOK_SCRIPT}
	touch ./${NOTEBOOK_SCRIPT}
	echo "#${SHELL}" >> ./${NOTEBOOK_SCRIPT}
	echo "source ${ENV_FOLDER}/notebook-venv/bin/activate" >> ./${NOTEBOOK_SCRIPT}
	echo 'jupyter $$1' >> ./${NOTEBOOK_SCRIPT}
	chmod +x ./${NOTEBOOK_SCRIPT}

${ENV_FOLDER}/executable-venv:
	[[ -d ${ENV_FOLDER} ]] || mkdir ${ENV_FOLDER}
	#venv_folder = ${ENV_FOLDER}/notebook-venv
	#echo venv_folder
	python -m venv ${ENV_FOLDER}/executable-venv
	source ${ENV_FOLDER}/executable-venv/bin/activate; pip install --upgrade pip; pip install --upgrade setuptools; pip install -r requirements-executable.txt
	source ${ENV_FOLDER}/executable-venv/bin/activate; pip install -e ${ROOT_FOLDER}

python-bash-script: ${ENV_FOLDER}/executable-venv
	rm -f ./${EXECUTABLE_SCRIPT}
	touch ./${EXECUTABLE_SCRIPT}
	echo "#${SHELL}" >> ./${EXECUTABLE_SCRIPT}
	echo "source ${ENV_FOLDER}/executable-venv/bin/activate" >> ./${EXECUTABLE_SCRIPT}
	echo 'python $$1' >> ./${EXECUTABLE_SCRIPT}
	chmod +x ./${EXECUTABLE_SCRIPT}

install: python-bash-script

clean:
	rm -rf ./${ENV_FOLDER}/*
	rm -f ./${NOTEBOOK_SCRIPT}