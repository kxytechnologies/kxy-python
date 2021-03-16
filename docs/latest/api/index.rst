

.. meta::
	:description: Documentation of the RESTful API of KxY backend.
	:keywords: KXY AutoML, Lean AutoML, RESTful API.
	:http-equiv=content-language: en

===========
RESTful API
===========
For programming languages other than Python, we provide the following RESTful API access to the KxY infrastructure. Programmatic access to the KxY infrastructure is billed on a per-request basis and requires a valid API key and a valid payment method associated to your account. For pricing details visit https://www.kxy.ai/pricing/.


API Endpoint
------------
The current API endpoint is :code:`https://api.kxy.ai/v2/wk/`.



Uploading Your Data
-------------------
All analyses requests submitted to our RESTful API must include an identifier referring to a csv file you previously uploaded. To upload file, you must first request a signed upload url.


.. http:post:: /generate-signed-upload-url/
	
	Requests a signed URl to upload a data file. Only CSV files are supported at this point, and the first row should contain column names.

	:form file_identifier: An identifier that is characteristic of the content of the csv file to upload (e.g. a hash of its content).
	:form timestamp: The epoch timestamp at the time this request is issued.
	:reqheader Content-Type: :code:`application/json`
	:reqheader X-API-Key: :code:`<YOUR API KEY>`
	:status 400: Returned when the request fails.
	:status 200: Returned when you previously uploaded a file with the same identifier or when a signed url was generated for the upload.



	**Example Response Format I**: The file wasn't previously uploaded and a signed url was successfully generated.

	.. sourcecode:: http

		HTTP/1.1 200 OK
		Content-Type: application/json

		{
			"message": "Signed upload url successfully generated for identifier [********].",
			"presigned_url": {
				"url": "...",
				"fields": 
					{
						"acl": "...",
						"key": "...", 
						"signature": "...", 
						"policy": "...",
						"...": "..."
					}
			}
		}

	The file is then uploaded by issuing a POST request to the returned :code:`url`, using the data in :code:`fields`, and by appending :code:`.csv` to the identifier to form the file name.

	Check out :code:`kxy.api.data_transfer.upload_data` to see how this is done in Python.



	**Example Response II**: The file was previously uploaded (remember, the identifier should be characteristic of the file content in that changing the content should also change the identifier).

	.. sourcecode:: http

		HTTP/1.1 200 OK
		Content-Type: application/json

		{
			"message": "The file with identifier [********] was previously uploaded."
		}



	**Example Response III**: The request fails.

	.. sourcecode:: http

		HTTP/1.1 400 Bad Request
		Content-Type: application/json

		{
			"message": "Failed to generate signed upload url for file [********].csv."
		}


Data Valuation
--------------

.. http:post:: /data-valuation/
	
	Estimate the highest performance achievable in a supervised learning problem, regression or classification.

	:form file_identifier: The identifier of the CSV file containing the data, which you must have previously uploaded.
	:form target_column: The name of the column in the CSV file to contains true labels.
	:form problem_type: The type of supervised learning problem (:code:`classification` or :code:`regression`).
	:form timestamp: The epoch timestamp at the time this request is issued.
	:form job_id: (optional) The :code:`job_id` that was returned last time you issued the same request.
	:reqheader Content-Type: :code:`application/json`
	:reqheader X-API-Key: :code:`<YOUR API KEY>`
	:status 400: Returned when the request is missing at least one mandatory parameter.
	:status 401: Returned when you disable your API key from the KxY portal.
	:status 402: Returned when you have not provided a valid payment method in the KxY portal, or when we are unable to charge your account for the request.	
	:status 200: Returned when your request was successful and the response body contains a :code:`job_id` or analysis results.


	**Example Response Format I**: The request was successfully submitted, the backend is at work, but results are not yet available.

	.. sourcecode:: http

		HTTP/1.1 200 OK
		Content-Type: application/json

		{
			"job_id": "******",
		}


	You should store the returned :code:`job_id`, and use it to try again until the request returns results. Requests without a valid :code:`job_id` are billed a small eco-friendly fee **and** a bigger analysis fee, whereas requests with a :code:`job_id` that was previously returned by the API are only billed the eco-friendly fee.


	**Example Response Format II**: The request was successfully submitted, and results are available.

	.. sourcecode:: http

		HTTP/1.1 200 OK
		Content-Type: application/json

		{
			"job_id": "******",
			"r-squared": "*.**",
			"log-likelihood": "***",
			"rmse": "*****",
			"accuracy": "*.**"			 
		}




Model-Free Variable Selection
-----------------------------

.. http:post:: /variable-selection/

	Runs the model-free variable selection analysis. The first variable is the variable that explains the label the most, when used in isolation. The second variable is the variable that complements the first variable the most for predicting the label etc.


	:form file_identifier: The identifier of the CSV file containing the data, which you must have previously uploaded.
	:form target_column: The name of the column in the CSV file to contains true labels.
	:form problem_type: The type of supervised learning problem (:code:`classification` or :code:`regression`).
	:form timestamp: The epoch timestamp at the time this request is issued.
	:form job_id: (optional) The :code:`job_id` that was returned last time you issued the same request.
	:reqheader Content-Type: :code:`application/json`
	:reqheader X-API-Key: :code:`<YOUR API KEY>`
	:status 400: Returned when the request is missing at least one mandatory parameter.
	:status 401: Returned when you disable your API key from the KxY portal.
	:status 402: Returned when you have not provided a valid payment method in the KxY portal, or when we are unable to charge your account for the request.	
	:status 200: Returned when your request was successful and the response body contains a :code:`job_id` or analysis results.


	**Example Response Format I**: The request was successfully submitted, the backend is at work, but results are not yet available.

	.. sourcecode:: http

		HTTP/1.1 200 OK
		Content-Type: application/json

		{
			"job_id": "******",
		}


	You should store the returned :code:`job_id`, and use it to try again until the request returns results. Requests without a valid :code:`job_id` are billed a small eco-friendly fee **and** a bigger analysis fee, whereas requests with a :code:`job_id` that was previously returned by the API are only billed the eco-friendly fee.


	**Example Response Format II**: The request was successfully submitted, and results are available.

	.. sourcecode:: http

		HTTP/1.1 200 OK
		Content-Type: application/json

		{
			"job_id": "******",
			"selection_order": [1, 2, 3],
			"variable": ["best_var_1",  "best_var_2", "best_var_3"],
			"r-squared": ["highest_r_squared_with_var_1", "highest_r_squared_with_vars_12", "highest_r_squared_with_vars_123"],
			"log-likelihood": ["highest_log_lik_with_var_1", "highest_log_lik_with_vars_12", "highest_log_lik_with_vars_123"],
			"rmse": ["lowest_rmse_with_var_1", "lowest_rmse_with_vars_12", "lowest_rmse_with_vars_123"],
			"accuracy": ["highest_accuracy_with_var_1", "highest_accuracy_with_vars_12", "highest_accuracy_with_vars_123"]			 
		}



Data-Driven Improvability
-------------------------

.. http:post:: /data-driven-improvability/
	
	Estimate the potential performance boost that a set of new explanatory variables can bring about.

	:form file_identifier: The identifier of the CSV file containing the data, which you must have previously uploaded.
	:form target_column: The name of the column in the CSV file to contains true labels.
	:form problem_type: The type of supervised learning problem (:code:`classification` or :code:`regression`).
	:form timestamp: The epoch timestamp at the time this request is issued.
	:form new_variables: The list of column names to be used as new explanatory variables.
	:form job_id: (optional) The :code:`job_id` that was returned last time you issued the same request.
	:reqheader Content-Type: :code:`application/json`
	:reqheader X-API-Key: :code:`<YOUR API KEY>`
	:status 400: Returned when the request is missing at least one mandatory parameter.
	:status 401: Returned when you disable your API key from the KxY portal.
	:status 402: Returned when you have not provided a valid payment method in the KxY portal, or when we are unable to charge your account for the request.	
	:status 200: Returned when your request was successful and the response body contains a :code:`job_id` or analysis results.


	**Example Response Format I**: The request was successfully submitted, the backend is at work, but results are not yet available.

	.. sourcecode:: http

		HTTP/1.1 200 OK
		Content-Type: application/json

		{
			"job_id": "******",
		}


	You should store the returned :code:`job_id`, and use it to try again until the request returns results. Requests without a valid :code:`job_id` are billed a small eco-friendly fee **and** a bigger analysis fee, whereas requests with a :code:`job_id` that was previously returned by the API are only billed the eco-friendly fee.


	**Example Response Format II**: The request was successfully submitted, and results are available.

	.. sourcecode:: http

		HTTP/1.1 200 OK
		Content-Type: application/json

		{
			"job_id": "******",
			"r-squared-boost": "*.**",
			"log-likelihood-boost": "***",
			"rmse-reduction": "*****",
			"accuracy-boost": "*.**"			 
		}



Model-Driven Improvability
--------------------------

.. http:post:: /model-driven-improvability/
	
	Estimate the extent to which a trained supervised learning model may be improved in a model-driven fashion (i.e. without resorting to additional explanatory variables).

	:form file_identifier: The identifier of the CSV file containing the data, which you must have previously uploaded.
	:form target_column: The name of the column in the CSV file to contains true labels.
	:form problem_type: The type of supervised learning problem (:code:`classification` or :code:`regression`).
	:form timestamp: The epoch timestamp at the time this request is issued.
	:form prediction_column: The name of the column containing predictions of the trained supervised learning model.
	:form job_id: (optional) The :code:`job_id` that was returned last time you issued the same request.
	:reqheader Content-Type: :code:`application/json`
	:reqheader X-API-Key: :code:`<YOUR API KEY>`
	:status 400: Returned when the request is missing at least one mandatory parameter.
	:status 401: Returned when you disable your API key from the KxY portal.
	:status 402: Returned when you have not provided a valid payment method in the KxY portal, or when we are unable to charge your account for the request.	
	:status 200: Returned when your request was successful and the response body contains a :code:`job_id` or analysis results.


	**Example Response Format I**: The request was successfully submitted, the backend is at work, but results are not yet available.

	.. sourcecode:: http

		HTTP/1.1 200 OK
		Content-Type: application/json

		{
			"job_id": "******",
		}


	You should store the returned :code:`job_id`, and use it to try again until the request returns results. Requests without a valid :code:`job_id` are billed a small eco-friendly fee **and** a bigger analysis fee, whereas requests with a :code:`job_id` that was previously returned by the API are only billed the eco-friendly fee.


	**Example Response Format II**: The request was successfully submitted, and results are available.

	.. sourcecode:: http

		HTTP/1.1 200 OK
		Content-Type: application/json

		{
			"job_id": "******",
			"lost-r-squared": "*.**",
			"lost-log-likelihood": "***",
			"lost-rmse": "*****",
			"lost-accuracy": "*.**",
			"residual-r-squared": "*.**",
			"residual-log-likelihood": "***",
			"residual-rmse": "*****",		 
		}


	:code:`lost-*` metrics represent the performance irreversibly lost while training the supervised learning model. 

	They are defined as the difference between the highest performance that can be achieved when using the explanatory variables to predict the target column, and the highest performance that can be achieved when using model predictions as sole explanatory variable to predict the target column. When these metrics are close to zero, the trained model is optimal in that its predictions capture all the *juice* that was in explanatory variables (i.e. the target and the explanatory variables are conditionally independent given the model prediction). 


	For regression problems, :code:`residual-*` metrics correspond to the highest performance achievable when using explanatory variables to predict regression residuals. When these metrics are high, the trained regression model can be improved additively. When they are close to zero, the regression model is optimal in that errors it make cannot be reduced using the same explanatory variables that were used to train the model.



Model Explanation
-----------------
Model explanation is identical to :ref:`Model-Free Variable Selection`, except that rather than explaining true labels with explanatory variables, you should explain model predictions with explanatory variables.




