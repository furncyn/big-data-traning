#!/usr/bin/env python3

"""Clean comment text for easier parsing."""

from __future__ import print_function

import re
import string
import argparse

import sys
import json

__author__ = ""
__email__ = ""

# Depending on your implementation,
# this data may or may not be useful.
# Many students last year found it redundant.
_CONTRACTIONS = {
    "tis": "'tis",
    "aint": "ain't",
    "amnt": "amn't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hell": "he'll",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "ol": "'ol",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodys": "somebody's",
    "someones": "someone's",
    "somethings": "something's",
    "thatll": "that'll",
    "thats": "that's",
    "thatd": "that'd",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wedve": "wed've",
    "well": "we'll",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whodve": "whod've",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}

# You may need to write regular expressions.

def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing FOUR strings 
    1. The parsed text.
    2. The unigrams
    3. The bigrams
    4. The trigrams
    """
    # Replace new lines and tab characters with a single space
    text = re.sub(r'([\s+]\\n[\s+])', ' ', text)
    text = re.sub(r'(\\n)', ' ', text)
    
    # Remove URLs. Replace them with what is inside the [].
    text = re.sub(r'(\(http://\S+\)\.)', '.', text)
    text = re.sub(r'(\(https://\S+\)\.)', '.', text)
    text = re.sub(r'(http://\S+)', '', text)
    text = re.sub(r'(\(http://\S+\))', '', text)
    text = re.sub(r'(https://\S+)', '', text)
    text = re.sub(r'(\(https://\S+\))', '', text)
    text = re.sub(r'(\[)', '', text)         
    text = re.sub(r'(\])', '', text)
        
    # Remove the first slash of a raw reference of subreddit and user
    # (/r/subredditname -> r/subredditname and /u/someuser -> u/someuser)
    m = re.search('(/[r|u]\S+)', text)
    if m:
        text = re.sub(r'(/)', '', text, 1)

    # Split text on a single space. Remove all empty tokens after doing the split.
    tokens = re.split(' ', text)
    while ''  in tokens:
        tokens.remove('')
    
    # Separate all external punctuation into their own tokens, but maintain punctuation within words
    i = 0
    length = len(tokens)
    while i < length:
        matches = re.search('([!?;:.,]+)', tokens[i])
        if matches:
            mlen = len(matches[0])
            mstr = matches[0]
            k = i
            for j in range(mlen):
                tokens[i] = tokens[i].replace(mstr[j], '')
                k += 1
                tokens.insert(k, mstr[j])
                length += 1
            i = k
        i += 1
    
    # Remove all punctuations and convert to all lower case
    for i in range(length):
        tokens[i] = tokens[i].lower()
        matches = re.findall('([#$%&*+<=>@^`~\"\\\(\)\[\]\{\}])', tokens[i])
        for m in matches:
            tokens[i] = tokens[i].replace(m, '')

    # There was for some reason empty tokens again, so we will remove them
    while ''  in tokens:
        tokens.remove('')

    # Parsed comment
    parsed_text = ""
    for token in tokens:
        parsed_text += token + " "
    
    # Unigrams
    unigrams = ""
    i = 0
    length = len(tokens)
    while i < length:
        m = re.search('([!?;:.,])', tokens[i])
        if not m:
            unigrams += tokens[i] + " "
        i += 1
    
    # Bigrams
    bigrams = ""
    # Constructing a bigram-pair list
    i = 0
    length = len(tokens)
    bigrams_tokens = []
    while i < length - 1:
        bigrams_tokens.append([tokens[i], tokens[i+1]])
        i += 1
    # Ignore all pairs where at least one token contains punctuations.
    # Parse the rest into a string
    for pair in bigrams_tokens:
        m = re.search('([!?;:.,])', pair[0]) or re.search('([!?;:.,])', pair[1])
        if not m:
            bigrams += pair[0] + "_" + pair[1] + " "
    
    # Trigrams
    trigrams = ""
    i = 0
    trigrams_tokens = []
    while i < length - 2:
        trigrams_tokens.append([tokens[i], tokens[i+1], tokens[i+2]])
        i += 1
    for triplet in trigrams_tokens:
        m = re.search('([!?;:.,])', triplet[0]) or re.search('([!?;:.,])', triplet[1]) or re.search('([!?;:.,])', triplet[2])
        if not m:
            trigrams += triplet[0] + "_" + triplet[1] + "_" + triplet[2] + " "
    
    result = [parsed_text, unigrams, bigrams, trigrams]
    return result
    

if __name__ == "__main__":
    # This is the Python main function.
    # You should be able to run
    # python cleantext.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.

    # YOUR CODE GOES BELOW.
    filename = sys.argv[1]
    with open(filename, 'r') as fp:
        for line in fp:
            start_ind = line.find("\"body\"")
            end_ind = line.find("\",\"", start_ind+7)
            body = line[start_ind+8:end_ind]
            result = sanitize(body)
            print(result)
