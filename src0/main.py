"""
Building a Semantic Role Labelling system for Vietnamese
Python libary

main script

Running experiments
"""

import lib
from datetime import datetime

dataFile = 'data.xml'
listWordNameFile = 'wordList.txt'
listClusterFile = 'labelList.txt'
listFeatureName = ['voice', 'position', 'phrase type', 'function tag', 'path tree', 'head word', 'predicate', 'distance']
#listFeatureName = ['position', 'voice', 'path tree reduce', 'predicate', 'head word', 'sub category', 'distance', 'function tag']
listFeatureIdenName = ['phrase type', 'position', 'voice', 'path tree', 'predicate', 'head word']
listFeatureClassName = ['phrase type', 'position', 'voice', 'path tree', 'predicate', 'head word']
listLabelOriginal = [u'ArgM-CAU', u'Arg4', u'ArgM-GOL', u'ArgM-EXT', u'ArgM-ADV', u'ArgM-NEG', u'ArgM-LVB', u'ArgM-MNR', u'ArgM-ADJ', 'None', u'ArgM-DSP', u'ArgM-COM', u'ArgM-RES', u'ArgM-MOD', u'ArgM-I', u'ArgM-REC', u'ArgM-DIS', u'ArgM-DIR', u'ArgM-Partice', u'ArgM-PRD', u'Arg1', u'Arg2', u'Arg3', u'ArgM-LOC', u'ArgM-TMP', u'Arg0', u'ArgM-PRP']
listLabelReduce = [u'Arg0', u'Arg1', u'ArgM-ADV', u'ArgM-DIR', u'ArgM-DIS', u'ArgM-EXT', u'ArgM-LOC', u'ArgM-MNR' , u'ArgM-MOD', u'ArgM-NEG', u'ArgM-PRP', u'ArgM-TMP']
#listLabelReduce = [u'Arg0', u'Arg1', u'ArgM-TMP']
foldNumber = 10
numberElements = 200

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
# listIDAfterChunking, listTreeAfterChunking, listRelAfterChunking, listArgAfterChunking = lib.chunking(listID1Rel, listTree1Rel, listRel1Rel, listArg1Rel)
#listIDAfterChunking, listTreeAfterChunking, listRelAfterChunking, listArgAfterChunking = lib.chunking(listIDTotal, listTreeTotal, listRelTotal, listArgTotal)
listIDTotal, listTreeTotal, listRelTotal, listArgTotal = lib.filterData(listIDTotal, listTreeTotal, listRelTotal, listArgTotal)
listIDAfterChunking, listTreeAfterChunking, listRelAfterChunking, listArgAfterChunking = lib.chunking(listIDTotal, listTreeTotal, listRelTotal, listArgTotal)
#lib.omitUnderscore(listIDAfterChunking, listTreeAfterChunking, listRelAfterChunking, listArgAfterChunking)

print 'Getting Feature'
listLabel, listFeature, listCount = lib.getFeature(listIDAfterChunking, listTreeAfterChunking, listRelAfterChunking, listArgAfterChunking, listWordName, listCluster)

listLabel = lib.getListLabelReduce(listLabel, listLabelOriginal)

listLE, leLabel, listEncode = lib.getLabelEncoderParameter(listFeature, listLabel)
print 'Separating Data'
# groupInfo, groupListLabel, groupListFeature = lib.kFold(listID1Rel, listTree1Rel, listRel1Rel, listArg1Rel, listWordName, listCluster, foldNumber, listLabelOriginal)
groupInfo, groupListLabel, groupListFeature, listOfListNumArgPerSen = lib.kFold(listIDTotal, listTreeTotal, listRelTotal, listArgTotal, listWordName, listCluster, foldNumber, listLabelOriginal)

print 'Transforming Data'
listOfListFeatureTrain, listOfListFeatureTest, listOfListLabelTrain, listOfListLabelTest, listOfListNumArg, listOfListPredicateType = lib.crossValidation(groupListLabel, groupListFeature, listLE, leLabel, foldNumber, groupInfo)
print 'Running Algorithm'

listPrecisionChunking, listRecallChunking, listF1ScoreChunking, listPrecisionClassify1Stage, listRecallClassify1Stage, listF1ScoreClassify1Stage, listPrecision1Stage, listRecall1Stage, listF1Score1Stage, listOfListLabelPredict, listOfListLabelILP, listDensityMatrix, listOfListVariable, listPrecisionArg, listRecallArg, listF1ScoreArg = lib.crossValidationSVM1Stage(listOfListFeatureTrain, listOfListFeatureTest, listOfListLabelTrain, listOfListLabelTest, listFeatureName, foldNumber, listOfListNumArg, listEncode, listOfListNumArgPerSen, listOfListPredicateType)
# listPrecisionChunking, listRecallChunking, listF1ScoreChunking, listPrecisionIden, listRecallIden, listF1ScoreIden, listPrecisionClass, listRecallClass, listF1ScoreClass, listPrecision2Stage, listRecall2Stage, listF1Score2Stage = lib.crossValidationSVM2Stage(listOfListFeatureTrain, listOfListFeatureTest, listOfListLabelTrain, listOfListLabelTest, listFeatureIdenName, listFeatureClassName, foldNumber, listOfListNumArg, listEncode)
print 'Output'

F1Score = lib.output1Stage(listPrecisionChunking, listRecallChunking, listF1ScoreChunking, listPrecisionClassify1Stage, listRecallClassify1Stage, listF1ScoreClassify1Stage, listPrecision1Stage, listRecall1Stage, listF1Score1Stage, listPrecisionArg, listRecallArg, listF1ScoreArg)
# F1Score = lib.output2Stage(listPrecisionChunking, listRecallChunking, listF1ScoreChunking, listPrecisionIden, listRecallIden, listF1ScoreIden, listPrecisionClass, listRecallClass, listF1ScoreClass, listPrecision2Stage, listRecall2Stage, listF1Score2Stage)

endTime = datetime.now()
print "Running time: "
print (endTime - startTime)
