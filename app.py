import streamlit as st
from summa import summarizer
from newspaper import Article

def clean_text(text):
    return text.replace("\n\n", " ").replace("\n", "")


def parse(url):
    #opts, args = getopt.getopt(argv, '')

    #if len(argv) == 0:
        #print('No URL found', file=sys.stderr)

    try:
    	article = Article(url)
    	article.download()
    	article.parse()
    	return (clean_text(article.text))
    except:
    	text = "Oops! No content."
    	return text

def textrank(text,ratio):
	summarized_text = summarizer.summarize(text, ratio=ratio, language="english", split=True, scores=True)

	return summarized_text

def inputURL():
	url = st.text_input("Input URL", "",)
	return url

def main():
	st.write("""
		# Summarizer
	""")

	url = inputURL()

	text = st.text_area("Text", parse(url), height=500)
	#text = st.text_area("Text", "", height=500)

	ratio = st.slider("Summarization fraction", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

	summary = textrank(text,ratio)

	st.write("# Summary")

	for sentence, score in summary:
		st.write(sentence)

main()

