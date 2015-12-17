"""
Building a Semantic Role Labelling system for Vietnamese

export model script

Exporting model (writing parameters) into file.
"""

import lib
import pickle
from datetime import datetime
dataFile = 'data.xml'
listWordNameFile = 'wordList.txt'
listClusterFile = 'labelList.txt'
listFeatureName = ['voice', 'position', 'phrase type', 'function tag', 'path tree', 'head word', 'predicate', 'distance']
listFeatureIdenName = ['phrase type', 'position', 'voice', 'path tree', 'predicate', 'head word']
listFeatureClassName = ['phrase type', 'position', 'voice', 'path tree', 'predicate', 'head word']
listLabelOriginal = [u'ArgM-CAU', u'Arg4', u'ArgM-GOL', u'ArgM-EXT', u'ArgM-ADV', u'ArgM-NEG', u'ArgM-LVB', u'ArgM-MNR', u'ArgM-ADJ', 'None', u'ArgM-DSP', u'ArgM-COM', u'ArgM-RES', u'ArgM-MOD', u'ArgM-I', u'ArgM-REC', u'ArgM-DIS', u'ArgM-DIR', u'ArgM-Partice', u'ArgM-PRD', u'Arg1', u'Arg2', u'Arg3', u'ArgM-LOC', u'ArgM-TMP', u'Arg0', u'ArgM-PRP']
listLabelReduce = [u'Arg0', u'Arg1', u'ArgM-ADV', u'ArgM-DIR', u'ArgM-DIS', u'ArgM-EXT', u'ArgM-LOC', u'ArgM-MNR' , u'ArgM-MOD', u'ArgM-NEG', u'ArgM-PRP', u'ArgM-TMP']
foldNumber = 1

startTime = datetime.now()

print 'Running Program'
print 'Reading Data'
listSentence, listID, listCDATA = lib.readData(dataFile)
listTag, listWord = lib.convertData(listSentence)
listTagClone, listWordClone = lib.convertData(listSentence)
listTree = lib.dataToTree(listTagClone, listWordClone)
listWordName, listCluster = lib.readWordCluster(listWordNameFile, listClusterFile)

listRel, listArg = lib.readCDATA(listCDATA, listWord, listID)
listID1Rel, listTree1Rel, listRel1Rel, listArg1Rel = lib.collectTree1Rel(listID, listTree, listRel, listArg)
listIDExtractFromMutliRel, listTreeExtractFromMutliRel, listRelExtractFromMutliRel, listArgExtractFromMutliRel = lib.extractFromMultiRel(listID, listTree, listRel, listArg)
listIDTotal, listTreeTotal, listRelTotal, listArgTotal = lib.mergeData(listID1Rel, listTree1Rel, listRel1Rel, listArg1Rel, listIDExtractFromMutliRel, listTreeExtractFromMutliRel, listRelExtractFromMutliRel, listArgExtractFromMutliRel)

listIDAfterChunking, listTreeAfterChunking, listRelAfterChunking, listArgAfterChunking = lib.chunking(listIDTotal, listTreeTotal, listRelTotal, listArgTotal)

print 'Getting Feature'

listLabel, listFeature = lib.getFeature(listIDAfterChunking, listTreeAfterChunking, listRelAfterChunking, listArgAfterChunking, listWordName, listCluster)

listLabel = lib.getListLabelReduce(listLabel, listLabelOriginal)

listLE, leLabel, listEncode = lib.getLabelEncoderParameter(listFeature, listLabel)
groupInfo, groupListLabel, groupListFeature = lib.kFold(listIDTotal, listTreeTotal, listRelTotal, listArgTotal, listWordName, listCluster, foldNumber, listLabelOriginal)

print 'Transforming Data'
listOfListFeatureTrain, listOfListLabelTrain, listOfListNumArg = lib.crossValidationTotal(groupListLabel, groupListFeature, listLE, leLabel, foldNumber, groupInfo)

print 'Getting Parameter'
clf, enc = lib.crossValidationSVMTotal(listOfListFeatureTrain, listOfListLabelTrain, listFeatureName, foldNumber, listOfListNumArg, listEncode)

with open('model.pkl', 'wb') as output:
    pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
    
with open('feature.pkl', 'wb') as output:
    pickle.dump(listLE, output, pickle.HIGHEST_PROTOCOL)
    
with open('label.pkl', 'wb') as output:
    pickle.dump(leLabel, output, pickle.HIGHEST_PROTOCOL)
    
with open('enc.pkl', 'wb') as output:
    pickle.dump(enc, output, pickle.HIGHEST_PROTOCOL)

endTime = datetime.now()
print "Running time: "
print (endTime - startTime)