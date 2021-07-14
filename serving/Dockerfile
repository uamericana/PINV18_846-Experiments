# Webservice Dockerfile
# It needs access to the class_names.json file in the models directory, hence why it's not in the webservice directory

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /app

COPY webservice/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY webservice/app .

# Copies the class names definition into the image
ARG MODEL_CLASS_NAMES_FILE
ADD ${MODEL_CLASS_NAMES_FILE} class_names.json