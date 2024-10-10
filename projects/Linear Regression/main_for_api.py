import pandas as pandas
import numpy as numpy
from sklearn.linear_model import LinearRegression
import json
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Adjust as needed

def lambda_handler(event, context):

    logger.info("Executing Python Regression")
    logger.info(event)

    # Log the event for debugging
    print(f"Received event: {json.dumps(event)}")

    # event = json.loads(event)
    data = json.loads(event['body'])

    try:
        X = numpy.array(data['data']['x'])
        Y = numpy.array(data['data']['y'])

        # Print the data to verify
        print("Data Received:")
        print(X)
        print(Y)

        # Create a Linear Regression model
        model = LinearRegression()

        # Train the model
        model.fit(X, Y)

        # Make predictions
        Y_pred = model.predict(X)

        # Print the model parameters
        print(f"Slope (Coefficient): {model.coef_[0]}")
        print(f"Intercept: {model.intercept_}")

        logger.info("Slope")
        logger.info(model.coef_[0])

    except Exception as e:
            # Error Response, if exception is caught
            response = {
                'statusCode': 400,
                'body': json.dumps({
                    'predictions': [],
                    'errors': [str(e)]
                })
            }
            
            logger.info("Error")
            logger.info(str(e)  )

            return response

    # Success Response
    response = {
        'statusCode': 200,
        'body': json.dumps({
            'predictions': [],
            'errors': []
        })
    }

    return response


"""
LOCAL TESTING SECTION

### Remove below section from code which is sent upstream. This is for local testing...
# Retrieve Sample Data from JSON
with open('../Chat GPT__LTSM Prediction/supporting_files/sample_lambda_request.json') as sample_data:
    data = json.load(sample_data)

lambda_handler(data, "context")

"""
