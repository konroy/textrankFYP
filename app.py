import streamlit as st
#from summa import summarizer
import numpy as np
from newspaper import Article
from nltk.tokenize.punkt import PunktSentenceTokenizer
#import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
import networkx as nx
#import pydot as pd
#from pyvis.network import Network
import pandas as pd

def clean_text(text):
    return text.replace("\n\n", " ").replace("\n", "")

def parse(url):
    try:
    	article = Article(url)
    	article.download()
    	article.parse()
    	return (clean_text(article.text))
    except:
    	text = "Oops! No content."
    	return text

#opts, args = getopt.getopt(argv, '')

    #if len(argv) == 0:
        #print('No URL found', file=sys.stderr)

def inputURL():
	url = st.text_input("Input URL", "",)
	return url

def tokenize(text):
	doc_tokenizer = PunktSentenceTokenizer()
	sentences_list = doc_tokenizer.tokenize(text)
	#st.write(sentences_list)
	return sentences_list

def tdMatrix(sentence):
	cv = CountVectorizer()
	cv_matrix = cv.fit_transform(sentence)
	#st.write(cv_matrix)
	normal_matrix = TfidfTransformer().fit_transform(cv_matrix)
	#tfidf = TfidfVectorizer()
	#matrix = tfidf.fit_transform(sentence)
	#df_temp = pd.DataFrame(matrix.toarray(), columns = tfidf.vocabulary_)
	#df_T = df_temp.T
	#df_T.to_csv('tfidf.csv')
	#st.dataframe(df_temp.T)
	res_graph = normal_matrix * normal_matrix.T
	return res_graph

def nxGraph(graph):
	nx_graph = nx.from_scipy_sparse_matrix(graph)
	return nx_graph

def showGraph(nxgraph):
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot(nx.draw_circular(nxgraph))
	st.write("Number of Edges: {}".format(nxgraph.number_of_edges()))
	st.write("Number of Nodes: {}".format(nxgraph.number_of_nodes()))

def textrank(nx_graph,sentences_list):
	
	ranks = nx.pagerank(nx_graph)
	sentence_array = sorted(((ranks[i],s) for i, s in enumerate(sentences_list)), reverse=True)
	sentence_array = np.asarray(sentence_array)
	rank_max = float(sentence_array[0][0])
	rank_min = float(sentence_array[len(sentence_array) - 1][0])

	temp_array = []
	df_temp = pd.DataFrame(sentence_array, columns = ['Score', 'Sentence'])

	flag = 0
	if rank_max - rank_min == 0:
		temp_array.append(0)
		flag = 1

	if flag != 1:
		for i in range(0, len(sentence_array)):
			temp_array.append((float(sentence_array[i][0]) - rank_min) / (rank_max - rank_min))
			df_temp.at[i,'Score'] = temp_array[i]

	#df_temp = pd.DataFrame(sentence_array, columns = ['Score', 'Sentence'])
	#for i in range(0, len(sentence_array)):
		#df_temp.at[i,'Score'] = temp_array[i]
	st.header("Scoring")
	st.table(df_temp)
	#df_temp.to_csv('textrank.csv')

	threshold = (sum(temp_array) / len(temp_array)) + 0.2
	st.write("The threshold of the summarization is ",threshold)
	sentences_list = []
	if len(temp_array) > 1:
		for i in range(0, len(temp_array)):
			if temp_array[i] > threshold:
				sentences_list.append(sentence_array[i][1])
	else:
		sentences_list.append(sentence_array[0][1])
	
	return sentences_list
	#summary = " ".join(str(x) for x in sentences_list)
	#return summary
	
	#summarized_text = summarizer.summarize(text, ratio=ratio, language="english", split=True, scores=True)
	#return summarized_text

def main():
	st.title("Summarizer")

	url = inputURL()

	text = st.text_area("Text", parse(url), height=500)

	st.title("Results")
	
	
	#text = st.text_area("Text", "", height=500)

	#ratio = st.slider("Summarization fraction", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

	sentences_list = tokenize(text)

	res_graph = tdMatrix(sentences_list)
	#st.write(type(res_graph))
	nx_graph = nxGraph(res_graph)

	#nx_graph = nxGraph(graph)

	
	summary = textrank(nx_graph,sentences_list)
	#st.write(summary)

	st.header("Summary")
	for lines in summary:
		st.write(lines)

	st.header("Graph")
	showGraph(nx_graph)
	#for sentence, score in summary:
	#	st.write(sentence)

main()