# extractive-summarization
Extractive Text Summarization for COVID-19 Articles

This mini project was implemented in June 2020 for Introduction to Information Retrieval course in Bogazici University.

It implements a modified version of PageRank Algorithm (a.k.a [LexRank](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html)) to perform extractive text summarization on abstracts of COVID-19 articles in COVID-19 Open Research Dataset using relevance annotations provided by TREC-COVID Challenge. 

For this project, 3 topics are selected for summarization:
* Coronavirus origin
* How does coronavirus spread?
* How do people die from the coronavirus?

For each topic, top 10 most salient documents are determined and from those documents 20 sentences are selected as the summary of the topic.

-----------------

Python 3.7

Dependencies: csv, nltk, spaCy, json, numpy, pysbd, xml


You can install spaCy by the following commands:

pip: `pip install -U spacy`

OR

conda: `conda install -c conda-forge spacy`

Download spaCy English Language package:

`python -m spacy download en_core_web_sm`

For sentence boundary detection:
pip: `pip install pysbd`

----------------------
You need to download: 
* [Dataset (1.5GB)](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-04-10.tar.gz)
* List of [topics](https://ir.nist.gov/covidSubmit/data/topics-rnd1.xml) (also available in the repo)
* [Relevance judgements](https://ir.nist.gov/covidSubmit/data/qrels-rnd1.txt) (also available in the repo)


Program should be placed in the same directory with all dataset and auxilary files.

**Dataset files:**

qrels-rnd1.txt

04-10-mag-mapping.txt (can be found inside the dataset folder)

topics-rnd1.xml

**Auxilary files:**

lemmatized_dictionary.json (not available due to size concerns but will be generated once the program run)

vocabulary.txt

idf_vector.npy

Program can run without auxilary files and generate them during the execution. However, it takes a VERY long time since the corpus is quite big.
I strongly recommend AGAINST running the program without the auxilary files. It may take hours to finish idf vector extraction.
