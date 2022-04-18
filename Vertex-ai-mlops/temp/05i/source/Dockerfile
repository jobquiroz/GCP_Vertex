
FROM gcr.io/deeplearning-platform-release/tf-cpu.2-3
WORKDIR /
# Install Additional Packages
RUN pip install cloudml-hypertune
## Copies the trainer code to the docker image
COPY trainer /trainer
## Sets up the entry point to invoke the trainer
ENTRYPOINT ["python", "-m", "trainer.train"]
