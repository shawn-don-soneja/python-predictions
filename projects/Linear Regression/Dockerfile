# lambda base image for Docker from AWS
FROM public.ecr.aws/lambda/python:latest

# copy all code and lambda handler
COPY requirements.txt ./

# install packages
RUN python3 -m pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY main_for_api.py ./

# run lambda handler
CMD ["main_for_api.lambda_handler"]