# extractive-summarization
Extractive Text Summarization for COVID-19 Articles

Libraries used in the assignment
------------------------
import csv #read csv files
import nltk # word tokenization
import spacy #lemmatization and sentence boundary detection
import json # read and write json files
import os
import numpy as np # matrix operations
import string
import random # initialization of x vector in PageRank algorithm
from pysbd.utils import PySBDFactory #sentence boundary detection
import xml.etree.ElementTree as ET # to parse topic xml file
----------------------
Python version 3.7

You can install spaCy by the following commands:

pip: `pip install -U spacy`

OR

conda: `conda install -c conda-forge spacy`

Download spaCy English Language package:

`python -m spacy download en_core_web_sm`

For sentence boundary detection:
pip: `pip install pysbd`

----------------------
Program should be placed in same directory with all dataset and auxilary files.

Dataset files:
qrels-rnd1.txt
04-10-mag-mapping.txt
topics-rnd1.xml

Auxilary files:
lemmatized_dictionary.json
vocabulary.txt
idf_vector.npy

Program can run without auxilary files and generate them during the execution. However,it takes a VERY long time since the corpus is quite big.
I strongly recommend AGAINST running the program without the auxilary files. It may take hours to finish idf vector extraction.
