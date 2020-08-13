import numpy as np
import argparse
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import words
import pandas as pd
import textract


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    token = nltk.word_tokenize(doc)
    # add parts of speech to token
    tag = nltk.pos_tag(token)
    # convert nltk pos into wordnet pos
    nltk2wordnet = [(i[0], convert_tag(i[1])) for i in tag]
    # if there are no synsets in token, ignore, else put in a list
    output = [wn.synsets(i, z)[0] for i, z in nltk2wordnet if len(wn.synsets(i, z))>0]
    return output


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """

    largest_similarity_values = []
    for syn1 in s1:
        similarity_values  =[]
        for syn2 in s2:
            sim_val = wn.path_similarity(syn1, syn2)
            if sim_val is not None:
                similarity_values.append(sim_val)
        if len(similarity_values) != 0:
            largest_similarity_values.append(max(similarity_values))
    return sum(largest_similarity_values) / len(largest_similarity_values)


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", "--file1", type = str,
                        help="Enter a path to the doc file",
                        default = None)
    
    parser.add_argument("-f2", "--file2", type = str,
                        help="Enter a path to the doc file",
                       default = None)
    parser.add_argument("-t", "--threshold", type = float,
                        help="Enter the threshold similarity.",
                        default = .5)
    
    args = vars(parser.parse_args())
    
    file_path_1 = args["file1"]
    file_path_2 = args["file2"]
    
    doc1 = textract.process(file_path_1).decode("utf-8")
    text1 = [line.strip() for line in doc1.split('\n') if line != '']

    doc2 = textract.process(file_path_1).decode("utf-8")
    text2 = [line.strip() for line in doc2.split('\n') if line != '']

    output = []
    for line1 in text1:
        for line2 in text2:
            try:
                sim_score = document_path_similarity(line1,line2)
            except:
                continue
            output.append([line1,line2,sim_score])
            
    output.sort(key=lambda x: x[2], reverse = True)

    threshold = args["threshold"]
    desired_output = []
    for line1,line2,score in output:
        if score >threshold:
            desired_output.append([line1,line2,score])
    
    print("the number of lines above the similarity threshold are {}\n".format(len(desired_output)))
    print("the similar lines are: \n\n")
    for line1,line2,score in desired_output:
        print("{} \n{}\n---->{:.2f}%".format(line1, line2, score*100))
        print("--------------------------")