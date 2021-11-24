# from __future__ import unicode_literals

if __name__ == '__main__':
	# import logging
	# logging.basicConfig(level=logging.DEBUG)
	import numpy as np
	import pandas as pd
	import kxy
	from kxy.api import upload_data

	df = pd.DataFrame(np.random.randn(20000, 50))
	upload_data(df)


