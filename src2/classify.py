"""
Building a Semantic Role Labelling system for Vietnamese

classify script

Input: a bracketed syntactic tree and corresponding predicate.
Output: list constituent with Semantic role labels.
"""

#!/usr/bin/python
# -*- coding: utf8 -*-
import lib
import pickle
import sys
import numpy as np

def reform(ss):
	return ss.replace(',', ' ')

with open('model.pkl', 'rb') as input:
    clf = pickle.load(input)
    
with open('feature.pkl', 'rb') as input:
    listLE = pickle.load(input)
    
with open('label.pkl', 'rb') as input:
    leLabel = pickle.load(input)

with open('enc.pkl', 'rb') as input:
    enc = pickle.load(input)
    
listEncode = list(leLabel.classes_)

listWordNameFile = 'wordList.txt'
listClusterFile = 'labelList.txt'
listFeatureName = ['voice', 'position', 'phrase type', 'function tag', 'path tree', 'head word', 'predicate', 'distance']

# sentence = u' ( S-TTL ( NP-SUB ( N-H Đất )  ( A nghèo )  )   ( VP ( V-H trở mình )  )    ) '
# rel = u'trở mình'
#sentence = raw_input("Input Sentence: ").decode('utf8')
#rel = raw_input("Input Predicate: ").decode('utf8')
parser = argparse.ArgumentParser(description='This is a demo script.')
parser.add_argument('-s','--sentence', help='Sentence', required=True)
parser.add_argument('-p','--predicate', help='Predicate', required=True)
args = parser.parse_args()
sentence = args.sentence.decode('utf8')
rel = args.predicate.decode('utf8')

raw_tree = sentence
parse_tree = ''
first_brac = True
writable = True
for i in range(1, len(raw_tree) -1):
	if raw_tree[i] == '(' and (raw_tree[i+1] in '?.,!'):
		writable = False
	if writable:
		if raw_tree[i] in '()':
			parse_tree += ' '
		if raw_tree[i] != '_':
			parse_tree += raw_tree[i]
		else:
			parse_tree += ' '
		if raw_tree[i] in '()':
			parse_tree += ' '
	else:
		if raw_tree[i] == ')':
			writable = True

listID = ['00']
listSentence = []
listSentence.append(sentence.split())

listRel = [[rel]]
listArg = [[['', '']]]
listTag, listWord = lib.convertData(listSentence)
listTagClone, listWordClone = lib.convertData(listSentence)
listTree = lib.dataToTree(listTagClone, listWordClone)
listWordName, listCluster = lib.readWordCluster(listWordNameFile, listClusterFile)

listIDAfterChunking, listTreeAfterChunking, listRelAfterChunking, listArgAfterChunking = lib.chunking(listID, listTree, listRel, listArg)
listLabel, listFeature = lib.getFeature(listIDAfterChunking, listTreeAfterChunking, listRelAfterChunking, listArgAfterChunking, listWordName, listCluster)

listFeature = lib.labelEncoderData(listFeature, listLE)
listFeature = listFeature.astype(int)
listFeature = lib.convertToDataFrame(listFeature)
listFeature = listFeature.loc[:,listFeatureName]
listFeature = np.asarray(listFeature)
listFeatureSVM = enc.transform(listFeature)
listLabelPredict = clf.predict(listFeatureSVM)
listLabelPredictDecode = []
for item in listLabelPredict:
	listLabelPredictDecode.append(listEncode[item])
i = 0
for tree in listTreeAfterChunking:
	for leaf in tree:
		if leaf.word != rel:
			sys.stdout.write(reform(leaf.word).encode('utf8') + '\n')
			sys.stdout.write(listLabelPredictDecode[i] + '\n')
			i += 1
		else:
			sys.stdout.write(reform(leaf.word).encode('utf8') + '\n')
			sys.stdout.write('predicate\n')
