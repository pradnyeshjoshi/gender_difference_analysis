"""Blueprint for computing the baseline.
Use from command line:

[angelo@mangoni src]$ python baseline.py ./data/training ./output/
loading dataset...
setting up dummy pipeline...
performing cv on dummy pipeline...
Accuracy: 0.50 (+/- 0.00)
[angelo@mangoni src]$

"""
import sys
import datasets
import numpy as np
import pandas as pd
import logging
import os
import pprint
import argparse
import pipeline
from lxml.etree import tostring
from lxml.builder import E
from collections import defaultdict
from sklearn.metrics import accuracy_score as accuracy
from time import time
from glob import glob

# accepts inputDir and outputDir variables from command line
# parser = argparse.ArgumentParser(description='Run the pan17 pipeline')

# parser.add_argument('inputDir', metavar='input', type=str, nargs='+',
#                     help='input directory with the training data', default='')
# parser.add_argument('outputDir', metavar='output', type=str, nargs='?',
#                     help='output directory for the XML files', default='')
# args = parser.parse_args()


logging.basicConfig(level=logging.DEBUG)

# inputdir = args.inputDir[0]
# try:
#     outputdir = args.outputDir[0]
# except:
#     logging.info("Output dir not given: will not produce XML files")

def runbaseline():

	train_dir = '../../data/pan17/pan17-author-profiling-training-dataset-2017-03-10/'
	test_dir = '../../data/pan17/pan17-author-profiling-test-dataset-2017-03-16/'
	reddit_dir = '../../data/rawdata/'

	logging.basicConfig(filename="prediction_log.log", filemode="w")
	logging.info("loading train dataset...")
	df = datasets.load_pan17(train_dir)
	corpus = df.corpus
	corpus['text'] = corpus.text.apply(lambda x: '\n'.join(x))  # join tweets
	logging.info(corpus.shape)

	logging.info("loading test dataset...")
	df_test = datasets.load_pan17(test_dir)
	corpus_test = df_test.corpus
	corpus_test['text'] = corpus_test.text.apply(lambda x: '\n'.join(x))  # join tweets
	logging.info(corpus_test.shape)

	# load pipeline
	pipe = pipeline.pipeline

	# initialize dict to store scores
	scores = defaultdict(dict)
	# initialize dict to store predictions for xml output
	output = {auth: {'id': auth,
	                 'lang': lang} for auth, lang in zip(corpus.author,
	                                                     corpus.lang)}

    # train on three languages and test on the 4h
#    subset = corpus[corpus.lang.isin(['es', 'ar', 'pt'])]
#    test = corpus[corpus.lang == 'en']
#    for target in [subset.variety, subset.gender]:
#        print(target)
#        logging.info("testing pipeline for %s on %s" %
#                     (target.name, test.lang.iloc[0]))
#        pipe.fit(subset.text,  # X
#                 target)  # Y
#        predictions = pipe.predict(test.text)
#        # compute scores from predictions
#        scores[test.lang.iloc[0]][target.name] = accuracy(test[target.name],
#                                                          predictions)
#        # save predictions to output dict
#        for auth, pred in zip(subset.author, predictions):
#            output[auth][target.name] = pred
    
    # subset = corpus[corpus.lang.isin(['es', 'ar', 'pt'])]
    # test = corpus[corpus.lang == 'en']

	subset = corpus
	test = corpus_test
	subset = pd.concat([subset, test])
	subset = subset.sample(n=11400, replace=False)
	logging.info("sampled data shape " + str(subset.shape))

	logging.info("testing pipeline for gender")

	start_time = time()
	pipe.fit(subset.text,  # X
	         subset.gender)  # Y
	logging.info("model fit in " + str(time()-start_time) + " sec")

	try:
		logging.info("train accuracy on train+test data", pipe.score(subset['text'], subset['gender']))
	except:
		logging.info("error calculating accuracy")

	# predictions = pipe.predict(test.text)
	# # compute scores from predictions
	# scores[test.lang.iloc[0]]['gender'] = accuracy(test['gender'],
	#                                                   predictions)
	# print("test accuracy", scores[test.lang.iloc[0]]['gender'])

	logging.info("loading reddit dataset")
	reddit_post_list = [y for x in os.walk(reddit_dir) for y in glob(os.path.join(x[0], '*.csv')) if 'comment' not in y]
	reddit_comment_list = [y for x in os.walk(reddit_dir) for y in glob(os.path.join(x[0], '*.csv')) if 'comment' in y]
	for r in reddit_post_list + reddit_comment_list:
		logging.info("loading ", r)
		file_type = "post" if r in reddit_post_list else "comment"
		logging.info(file_type)
		df_reddit = datasets.load_reddit(r, type=file_type)
		logging.info("predict on ", r)
		pred = pipe.predict_proba(df_reddit.corpus.text)
		df_reddit.corpus['predicted_gender'] = pred
		df_reddit.corpus.to_csv(r, index=False, index_label=False)


	# save predictions to output dict
	# for auth, pred in zip(test.author, predictions):
	#     output[auth]['gender'] = pred

	# if outputdir is defined then:
	# shape output as xml and print to file in outputdir
	# otherwise we are done
	# try:
	#     for entry in output:
	#         out = tostring(E.author(id=output[entry]['id'],
	#                                 lang=output[entry]['lang'],
	#                                 # variety=output[entry]['variety'],
	#                                 gender=output[entry]['gender']),
	#                        pretty_print=True)
	#         with open(os.path.join(outputdir, entry), 'wb+') as f:
	#             f.write(out)
	# except:
	#     logging.info("No output dir given. Done.")
	#



	logging.info(pprint.pprint(scores))
	logging.info("Averaged accuracy %0.2f" %
	             np.mean(list((list(x.values()) for x in scores.values()))))


if __name__ == "__main__":
    runbaseline()
