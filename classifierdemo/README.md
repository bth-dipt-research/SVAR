# About

This is a simple demonstrator of the SetFit classifier. The trained models are available on HuggingFace. To be able to access the private models, one has to set the `HF_TOKEN`environment variable.

Run the demonstrator:

```
export HF_TOKEN=$SECRET_TOKEN && python ui/app.py
```


To build and run the demonstrator as a docker container:

```
docker build -t trvinfraclassifier:latest .
docker run --rm -e HF_TOKEN=$SECRET_TOKEN trvinfraclassifier
```

