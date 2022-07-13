# import the necessary libraries
import nltk
import string
import re

def text_lowercase(text):
	return text.lower()

input_str = "Hey, did you know that the summer break is coming? Amazing right !! It's only 5 more days !!"
text_lowercase(input_str)

# Remove numbers
def remove_numbers(text):
	result = re.sub(r'\d+', '', text)
	return result

input_str = "There are 3 balls in this bag, and 12 in the other one."
remove_numbers(input_str)

# import the inflect library
import inflect
p = inflect.engine()

# convert number into words
def convert_number(text):
	# split string into list of words
	temp_str = text.split()
	# initialise empty list
	new_string = []

	for word in temp_str:
		# if word is a digit, convert the digit
		# to numbers and append into the new_string list
		if word.isdigit():
			temp = p.number_to_words(word)
			new_string.append(temp)

		# append the word as it is
		else:
			new_string.append(word)

	# join the words of new_string to form a string
	temp_str = ' '.join(new_string)
	return temp_str

input_str = 'There are 3 balls in this bag, and 12 in the other one.'
convert_number(input_str)


# Remove whitespaces:

# remove punctuation
def remove_punctuation(text):
	translator = str.maketrans('', '', string.punctuation)
	return text.translate(translator)

input_str = "Hey, did you know that the summer break is coming? Amazing right !! It's only 5 more days !!"
remove_punctuation(input_str)

# remove whitespace from text
def remove_whitespace(text):
	return " ".join(text.split())

input_str = " we don't need the given questions"
remove_whitespace(input_str)


# Remove default stopwords:

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# remove stopwords function
def remove_stopwords(text):
	stop_words = set(stopwords.words("english"))
	word_tokens = word_tokenize(text)
	filtered_text = [word for word in word_tokens if word not in stop_words]
	return filtered_text

example_text = "This is a sample sentence and we are going to remove the stopwords from this."
remove_stopwords(example_text)


# Stemming:

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer = PorterStemmer()

# stem words in the list of tokenized words
def stem_words(text):
	word_tokens = word_tokenize(text)
	stems = [stemmer.stem(word) for word in word_tokens]
	return stems

text = 'data science uses scientific methods algorithms and many types of processes'
stem_words(text)

# Lemmatization:

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
# lemmatize string
def lemmatize_word(text):
	word_tokens = word_tokenize(text)
	# provide context i.e. part-of-speech
	lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens]
	return lemmas

text = 'data science uses scientific methods algorithms and many types of processes'
lemmatize_word(text)


# Part of Speech Tagging:

from nltk.tokenize import word_tokenize
from nltk import pos_tag

# convert text into word_tokens with their tags
def pos_tagging(text):
	word_tokens = word_tokenize(text)
	return pos_tag(word_tokens)

pos_tagging('You just gave me a scare')

# download the tagset
nltk.download('tagsets')

# extract information about the tag
nltk.help.upenn_tagset('NN')

# Chunking:
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# define chunking function with text and regular
# expression representing grammar as parameter
def chunking(text, grammar):
	word_tokens = word_tokenize(text)

	# label words with part of speech
	word_pos = pos_tag(word_tokens)

	# create a chunk parser using grammar
	chunkParser = nltk.RegexpParser(grammar)

	# test it on the list of word tokens with tagged pos
	tree = chunkParser.parse(word_pos)
	
	for subtree in tree.subtrees():
		print(subtree)
	tree.draw()
	
sentence = 'the little yellow bird is flying in the sky'
grammar = "NP: {<DT>?<JJ>*<NN>}"
chunking(sentence, grammar)

# Named Entity Recognition:
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

def named_entity_recognition(text):
	# tokenize the text
	word_tokens = word_tokenize(text)

	# part of speech tagging of words
	word_pos = pos_tag(word_tokens)

	# tree of word entities
	print(ne_chunk(word_pos))

text = 'Bill works for GeeksforGeeks so he went to Delhi for a meetup.'
named_entity_recognition(text)
