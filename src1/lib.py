#!/usr/bin/python
# -*- coding: utf8 -*-
"""
Building a Semantic Role Labelling system for Vietnamese
Python libary
"""
import xml.etree.ElementTree as ET
import numpy as np
import operator
from ete2 import Tree
from collections import Counter
import string, copy
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
from sklearn import mixture
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from pulp import *

def readData(filename):
    """Return list of sentences, ID and CDATA 
    after reading data of xml formatted file "filename"
    """
    listID = []
    listSentence = []
    listCDATA = []
    tree = ET.parse(filename)
    root = tree.getroot()
    for file in root:
        for sentence in file:
            listID.append(sentence.attrib['id'])
            sen = ''
            CDATA = []
            for vtb in sentence:
                if vtb.tag == 'vtb':
                    sen += vtb.text + ' '
                elif vtb.tag == 'srl':
                    for rel in vtb:
                        for pb in rel:
                            if pb.text.find('CDATA') != -1:
                                CDATA.append(pb.text)
            listSentence.append(sen.split())
            listCDATA.append(CDATA)
    return listSentence, listID, listCDATA

def convertData(listSentence):
    """Return list of words and tags (brackets and syntactic labels)
    """
    listTag = []
    listWord = []
    for record in listSentence:
        temp = []
        recordNew = []
        recordWord = []
        backword = []
        backbackword = []
        for word in record:
            if word == '(':
                if(backword != ')'):
                    recordNew.append(word)
                else:
                    recordNew.pop()
                    recordNew.append(',')
            elif(word == ")"):
                recordNew.append(temp.pop())
                recordNew.append(word)
            else:
                if (backword == '(') or (backword == ')'):
                    temp.append(word)
                elif (backbackword == '(') or (backbackword == ')'):
                    recordWord.append(word)
                else:
                    tempWord = recordWord.pop()
                    tempWord = tempWord + ' ' + word
                    recordWord.append(tempWord)
            backbackword = backword
            backword = word
        listTag.append(recordNew)
        listWord.append(recordWord)
    return listTag, listWord

def dataToTree(listTag, listWord):
    """return list of bracketed syntactic trees
    Each node's name is a syntactic label.
    Each leaf has a feature named word which stores 1 word of sentence
    """
    listTree = []
    for i in range(len(listTag)):
        tempString = ''.join(listTag[i])
        tempString = tempString + ';'
        tempTree = Tree(tempString, format = 1)
        for leaf in tempTree:
            leaf.add_features(word = listWord[i].pop(0))
        listTree.append(tempTree)
    return listTree

def reformWord(ss):
    """Return string after removing punctuation
    """
    punc = u':,.;/!?\"-“'
    words = ss.split(u' ')
    temp = ''
    for word in words:
        for p in punc:
            word = word.replace(p, '')
        temp += word + ' '
    words = temp.split()
    temp = ''
    for word in words:
        temp += word + ' '
    return temp.strip()

def isPhraseType(ss):
    """Return True if string "ss" is a phrase label
    		  False if not
    """
    specialTag = ['MDP', 'UCP', 'WHAP', 'WHNP', 'WHPP', 'WHRP', 'WHVP', 'WHXP']
    for tag in specialTag:
        if ss.find(tag) != -1:
            return True
    if len(ss) > 1:
        if ss[1] == 'P':
            return True
    return False

def isPP(ss):
    """Return True if string "ss" is a adposition phrase label
    		  False if not
    """
    if len(ss) > 1:
        if ss[0] == 'P' and ss[1] == 'P':
            return True
    return False

def isSType(ss):
    """Return True if string "ss" is a sentence label
    		  False if not
    """
    if len(ss) > 0:
        if ss[0] == 'S':
            return True
    return False

def readCDATA(listCDATA, listWord, listID):
    """Return list of relations and arguments in the sentence
    after reading CDATA
    """
    listArg = []
    listRel = []
    for i in range(len(listCDATA)):
        sentence = ''
        for word in listWord[i]:
            sentence += word + ' '		
        sentence = reformWord(sentence)
        aa = []
        rr = []
        for CDATA in listCDATA[i]:
            args = []
            rels = []
            end = 0
            while True:
                arg = []
                beg1 = CDATA.find('Arg', end)
                beg2 = CDATA.find('ARG', end)
                if beg1 == -1:
                    beg1 = 999999
                if beg2 == -1:
                    beg2 = 999999
                beg = min(beg1, beg2)
                if beg == 999999:
                    break
                end = CDATA.find('<', beg)
                pos = CDATA.find(':', beg, end)
                arg.append(CDATA[beg:pos])
                arg.append(reformWord(CDATA[pos+2:end]))
                args.append(arg)
            end = 0
            while True:
                beg1 = CDATA.find('Rel', end)
                beg2 = CDATA.find('REL', end)
                beg3 = CDATA.find('rel', end)
                if beg1 == -1:
                    beg1 = 999999
                if beg2 == -1:
                    beg2 = 999999
                if beg3 == -1:
                    beg3 = 999999
                beg = min(beg1, beg2, beg3)
                if beg == 999999:
                    break
                end = CDATA.find('<', beg)
                pos = CDATA.find(':', beg, end)
                rels.append(reformWord(CDATA[pos+2:end]))
            isExist = True
            for arg in args:
                if sentence.find(arg[1]) == -1:
                    isExist = False
                    break
            if isExist:
                for arg in args:
                    aa.append(arg)
                for rel in rels:
                    rr.append(rel)
        listArg.append(aa)
        listRel.append(rr)
    return listRel, listArg

def collectTree1Rel(listID, listTree, listRel, listArg):
    """return list of sentences (ID, tree, list of relations and list of arguments) 
    which have 1 relation
    """
    newListID = []
    newListTree = []
    newListRel = []
    newListArg = []
    for i in range(len(listTree)):
        if len(listRel[i]) == 1 and len(listArg) > 0:
            newListID.append(listID[i])
            newListTree.append(listTree[i])
            newListRel.append(listRel[i])
            newListArg.append(listArg[i])
    return newListID, newListTree, newListRel, newListArg

def collect(node):
    """collect words at leaves of subtree rooted at "node"
    create a constituent and add feature "headWord" which stores head word type.
    """
    leaves = node.get_leaves()
    headWordType = leaves[0].name
    temp = ''
    for leaf in leaves:
        temp += leaf.word + ','
    temp = temp.rstrip(',')
    node.add_features(word = temp)
    node.add_features(headWord = headWordType)
    for child in node.get_children():
        child.detach()

def phraseType(tag):
    """Return phrase type
    """
    return tag.split('-')[0]

def process(node):
    """process creating constituent
    """
    children = node.get_children()
    if len(children) > 1 and isPhraseType(children[0].name):
        same = True
        for child in children:
            if phraseType(child.name) != phraseType(children[0].name):
                same = False
                break
        diff = True
        n = 0
        for child in children:
            if n == 0:
                n += 1
                continue
            if child.name == children[0].name:
                diff = False
        if same and diff:
            for child in children:
                collect(child)
        else:
            collect(node)
    else:
        collect(node)

def getLeavesPredicate(tree, predicate, id):
    """Return represented node of 1 long predicate (words in many leaves)
    return None if not found
    """
    leaves = tree.get_leaves()
    for i in range(len(leaves)):
        if predicate.find(leaves[i].word) == 0:
            temp = ''
            temp += leaves[i].word
            for j in range(i+1, len(leaves)):
                if predicate.find(temp + ' ' + leaves[j].word) == 0:
                    temp += ' ' + leaves[j].word
                else:
                    if predicate == temp:
                        for k in range(i+1, j):
                            ancestor = leaves[i].get_common_ancestor(leaves[k])
                        newNode = ancestor.add_child(name = ancestor.name)
                        newNode.add_features(word = predicate)
                        for k in range(i, j):
                            leaves[k].detach()
                        while True:
                            cnt = 0
                            for leaf in tree:
                                try:
                                    a = leaf.word
                                except:
                                    leaf.detach()
                                    cnt += 1
                            if cnt == 0:
                                break
                        return newNode
                    else:
                        temp = ''
            if predicate == temp:
                for k in range(i+1, len(leaves)):
                    ancestor = leaves[i].get_common_ancestor(leaves[k])
                newNode = ancestor.add_child(name = ancestor.name)
                newNode.add_features(word = predicate)
                for k in range(i, len(leaves)):
                    leaves[k].detach()
                while True:
                    cnt = 0
                    for leaf in tree:
                        try:
                            a = leaf.word
                        except:
                            leaf.detach()
                            cnt += 1
                    if cnt == 0:
                        break
                return newNode
    return None

def chunking(listID, listTree, listRel, listArg):
    """Return list of tree after applying Constituent Extraction Algorithm
    with list of ID, realation and corresponding list of arguments
    """
    newListID = []
    newListTree = []
    newListRel = []
    newListArg = []
    for i in range(len(listTree)):
        id = listID[i]
        tree = listTree[i]
        rels = listRel[i]
        args = listArg[i]
        index = 0
        for rel in rels:
            if rel == '':
                continue
            if len(tree.search_nodes(word = rel)) > 0:
                for j in range(len(tree.search_nodes(word = rel))):
                    tempTree = tree.copy()
                    currentNode = tempTree.search_nodes(word = rel)[j]
                    while currentNode != tempTree:
                        for sister in currentNode.get_sisters():
                            process(sister)
                        currentNode = currentNode.up
                    newListID.append(id)
                    newListTree.append(tempTree)
                    newListRel.append(rel)
                    newListArg.append(args)
            else:
                tempTree = tree.copy()
                currentNode = getLeavesPredicate(tempTree, rel, id)
                if currentNode == None:
                    continue
                while currentNode != tempTree:
                    for sister in currentNode.get_sisters():
                        process(sister)
                    currentNode = currentNode.up
                newListID.append(id)
                newListTree.append(tempTree)
                newListRel.append(rel)
                newListArg.append(args)
    return newListID, newListTree, newListRel, newListArg

def unique(seq):
    """Return list of unique elements in list "seq"
    """
    seen = set()
    seen_add = seen .add
    return [x for x in seq if not (x in seen or seen_add(x))]

def filterRel(listID, listTree, listRel, listArg):
    """Return 2 list of sentences (ID, tree, list of relations and arguments)
    after dividing 2 groups:
    + sentences which have verbal relation (except "là")
    + remaining sentences
    """
    listID1 = []
    listTree1 = []
    listRel1 = []
    listArg1 = []
    listID2 = []
    listTree2 = []
    listRel2 = []
    listArg2 = []
    for i in range(len(listTree)):
        id = listID[i]
        tree = listTree[i]
        rel = listRel[i]
        args = listArg[i]
        node = tree.search_nodes(word = rel)[0]
        if node.name[0] == u'V':
            if rel == u'là':
                listID2.append(id)
                listTree2.append(tree)
                listRel2.append(rel)
                listArg2.append(args)
            else:
                listID1.append(id)
                listTree1.append(tree)
                listRel1.append(rel)
                listArg1.append(args)
        else:
            listID2.append(id)
            listTree2.append(tree)
            listRel2.append(rel)
            listArg2.append(args)
    return listID1, listTree1, listRel1, listArg1, listID2, listTree2, listRel2, listArg2

def isSame(word, Arg):
    """Return True if content of constituent and content of argument are same
              False if not.
    """
    word1 = word.replace(',', ' ')
    listWord = word1.strip().split()
    listArg = Arg.strip().split()
    if len(listWord) != len(listArg):
        return False
    for i in range(len(listArg)):
        s = reformWord(listArg[i]) 
        t = reformWord(listWord[i])
        if s != t:
            return False
    return True

def evaluateChunking(listID, listTree, listRel, listID1, listArg1):
    """Return list ID and arguments which can't catch by Constituent Extraction
    print number of constituents which are arguments and number of arguments
    """
    count = 0
    numberConstituent = 0
    numberArg = 0
    newListID = []
    newListTree = []
    newListRel = []
    newListArg = []
    listTag = []
    listUniqueID = unique(listID)
    print 'number sentences: ' + str(len(listUniqueID))
    for i in range(len(listUniqueID)):
        id = listUniqueID[i]
        args = listArg1[listID1.index(id)]
        numberArg += len(args)
        for arg in args:
            done = False
            for j in [x for x in range(len(listID)) if listID[x] == id]:
                for leaf in listTree[j]:
                    if isSame(leaf.word, arg[1]):
                        count += 1
                        done = True
                        break
            if not done:
                newListID.append(id)
                newListArg.append(arg)
    print 'number constituent be argument: ' + str(count)
    print 'number argument: ' + str(numberArg)
    return newListID, newListArg

def reformTag(ss):
    """Removing digits in syntactic labels.
    """
    for d in string.digits:
        ss = ss.replace(d, '')
    return ss

def reformTag1(ss):
    """Return phrase type. (replacing sentence labels by "S")
    """
    for d in string.digits:
        ss = ss.replace(d, '')
    s = ss.split('-')
    temp = ''.join(s[0])
    # if len(s) >= 2:
    #     if s[1] == 'H':
    #         temp += '-'
    #         temp += s[1]
    if temp[0] == 'S':
        temp = 'S'
    if temp == 'VY-H':
        # temp = 'Vy-H'
        temp = 'Vy'

    return temp

def getPath(tree, predicate):
    """Return list of Parse tree path and distance (features)
    """
    predicateNode = tree.search_nodes(word = predicate)[0]
    listPath = []
    listDistance = []
    for leaf in tree:
        if leaf.word != predicate:
            ancestor = predicateNode.get_common_ancestor(leaf)
            path = []
            currentNode = leaf
            while currentNode != ancestor:
                path.append(reformTag1(currentNode.name))
                path.append('1')
                currentNode = currentNode.up
            path.append(reformTag1(ancestor.name))
            temp = []
            currentNode = predicateNode
            while currentNode != ancestor:
                temp.append(reformTag1(currentNode.name))
                temp.append('0')
                currentNode = currentNode.up
            path.extend(temp[::-1])
            listPath.append(''.join(path))
            listDistance.append(len(path)/2+1)
    return listPath, listDistance

def getHalfPath(listPath):
    """Return list of Patial Parse tree path (feature)
    """
    listHalfPath = []
    for path in listPath:
        temp = ''
        for c in path:
            if c != '0':
                temp += c
            else:
                break
        listHalfPath.append(temp)
    return listHalfPath

def getPhraseType(tree, predicate):
    """Return list of phrase type (feature)
    """
    listPhraseType = []
    for leaf in tree:
        if leaf.word != predicate:
            listPhraseType.append(reformTag1(leaf.name))
    return listPhraseType

def getTagFunction(ss):
    """Return function tag of syntactic label.
    """
    for d in string.digits:
        ss = ss.replace(d, '')
    s = ss.split('-')
    if len(s) == 2:
        return s[1]
    elif len(s) > 2:
        if s[2] == '':
            return s[1]
        else:
            if s[2] == 'TPC' or s[2] == 'SUB':
                return s[2]
            return s[1]
    return 'None'

def getFunctionType(tree, predicate):
    """Return list of Function tag. (feature)
    """
    listFunctionType = []
    for leaf in tree:
        if leaf.word != predicate:
            listFunctionType.append(getTagFunction(leaf.name))
    return listFunctionType

def getPosition(tree, predicate):
    """Return list of Position (feature)
    """
    listPosition = []
    pos = 0
    for leaf in tree:
        if leaf.word != predicate:
            listPosition.append(pos)
        else:
            pos = 1-pos
    return listPosition

def getVoice(tree, predicate):
    """Return list of voice (feature)
    """
    if len(tree.search_nodes(word = u'bị')) > 0:
        for node in tree.search_nodes(word = u'bị'):
            if node.name == 'V-H':
                for sister in node.get_sisters():
                    if sister.name == 'SBAR':
                        return 0
    if len(tree.search_nodes(word = u'được')) > 0:
        for node in tree.search_nodes(word = u'được'):
            if node.name == 'V-H':
                for sister in node.get_sisters():
                    if sister.name == 'SBAR':
                        return 0
    return 1

def getHeadWord(tree, predicate):
    """Return list of Head word (feature)
    """
    listHeadWord = []
    for leaf in tree:
        if leaf.word != predicate:
            listHeadWord.append(leaf.word.split(',')[0].strip())
    return listHeadWord

def getSubCategorization(tree, predicate):
    """Return list of subcategories (feature)
    """
    listSubCategorization = []
    predicateNode = tree.search_nodes(word = predicate)[0]
    for leaf in tree:
        if leaf.name != predicate:
            ancestor = predicateNode.up
            subtree = ancestor.copy()
            for node in subtree.traverse("postorder"):
                node.name = reformTag1(node.name)
            listSubCategorization.append(subtree.write(format = 8))
    return listSubCategorization

def reformLabel(tag):
    """Reform argument label
    """
    if tag[3].isdigit():
        return 'Arg' + tag[3]
    return 'Arg' + tag[3:]

def readWordCluster(filename1, filename2):
    """Return list of word clusters
    """
    f = open(filename1, 'r')
    listWordName = f.readlines()[0]
    listWordName = listWordName.split()
    listWordNameNew = []
    f.close()
    for word in listWordName:
        wordTemp = word.split('_')
        wordTemp = ' '.join(wordTemp)
        wordTemp = wordTemp.decode('utf-8', 'ignore')
        listWordNameNew.append(wordTemp)
    f = open(filename2, 'r')
    listCluster = f.readlines()[0]
    listCluster = listCluster.split()
    f.close()
    return listWordNameNew, listCluster

def getFeature(listID, listTree, listRel, listArg, listWordName, listCluster):
    """Return list of label and corresponding list of features 
    """
    listFeature = []
    listLabel = []
    listCount = []
    for i in range(len(listTree)):
        id = listID[i]
        tree = listTree[i]
        rel = listRel[i]
        args = listArg[i]
        listPath, listDistance = getPath(tree, rel)
        listHalfPath = getHalfPath(listPath)
        listPhraseType = getPhraseType(tree, rel)
        listPosition = getPosition(tree, rel)
        voice = getVoice(tree, rel)
        listHeadWord = getHeadWord(tree, rel)		
        listSubCategorization = getSubCategorization(tree, rel)
        listFunctionType = getFunctionType(tree, rel)
        predicateNode = tree.search_nodes(word = rel)[0]
        i = 0
        for leaf in tree:
            if leaf.word == rel:
                continue
            done = False
            feature = []

            found = False
            for j in range(len(listWordName)):
                if rel.lower() == listWordName[j]:
                    relNew = listCluster[j]
                    found = True
                    break
            if not found:
                relNew = '128'
            #relNew = rel
            feature.append(relNew)
            feature.append(listPath[i])
            feature.append(listPhraseType[i])
            feature.append(listPosition[i])
            feature.append(voice)
            # found = False
            # for j in range(len(listWordName)):
            # 	if(listHeadWord[i] == listWordName[j]):
            # 		listHeadWordNew = listCluster[j]
            # 		found = True
            # 		break
            # if not found:
            # 	listHeadWordNew = '128'
            listHeadWordNew = listHeadWord[i].lower()
            feature.append(listHeadWordNew)
            feature.append(listSubCategorization[i])
            feature.append(listHalfPath[i])
            feature.append(listDistance[i])
            feature.append(leaf.headWord)
            feature.append(listFunctionType[i])
            # feature.append(predicateNode.name)
            feature.append(phraseType(predicateNode.name))
            listFeature.append(feature)
            i += 1
            for arg in args:
                if isSame(leaf.word, arg[1]):
                    if reformLabel(arg[0]) == 'ArgM':
                        # print id
                        pass
                    listLabel.append(reformLabel(arg[0]))
                    done = True
                    break
            if not done:
                listLabel.append('None')
        listCount.append(i)
    return listLabel, listFeature, listCount


def getFeature1(listID, listTree, listRel, listArg, listWordName, listCluster):
    listFeature = []
    listLabel = []
    for i in range(len(listTree)):
        id = listID[i]
        tree = listTree[i]
        rel = listRel[i]
        args = listArg[i]
        listPath, listDistance = getPath(tree, rel)
        listHalfPath = getHalfPath(listPath)
        listPhraseType = getPhraseType(tree, rel)
        listPosition = getPosition(tree, rel)
        voice = getVoice(tree, rel)
        listHeadWord = getHeadWord(tree, rel)		
        listSubCategorization = getSubCategorization(tree, rel)
        listFunctionType = getFunctionType(tree, rel)
        i = 0
        for leaf in tree:
            if leaf.word == rel:
                continue
            done = False
            feature = []
            # found = False
            # for j in range(len(listWordName)):
            # 	if rel == listWordName[j]:
            # 		relNew = listCluster[j]
            # 		found = True
            # 		break
            # if not found:
            # 	relNew = '128'
            relNew = rel
            feature.append(relNew)
            feature.append(listPath[i])
            feature.append(listPhraseType[i])
            feature.append(listPosition[i])
            feature.append(voice)
            # found = False
            # for j in range(len(listWordName)):
            # 	if(listHeadWord[i] == listWordName[j]):
            # 		listHeadWordNew = listCluster[j]
            # 		found = True
            # 		break
            # if not found:
            # 	listHeadWordNew = '128'
            listHeadWordNew = listHeadWord[i]
            feature.append(listHeadWordNew)
            feature.append(listSubCategorization[i])
            feature.append(listHalfPath[i])
            feature.append(listDistance[i])
            feature.append(leaf.headWord)
            feature.append(listFunctionType[i])
            listFeature.append(feature)
            i += 1
            for arg in args:
                if isSame(leaf.word, arg[1]):
                    if reformLabel(arg[0]) == 'ArgM':
                        # print id
                        pass
                    listLabel.append(reformLabel(arg[0]))
                    done = True
                    break
            if not done:
                listLabel.append('None')
    return listLabel, listFeature

def writeFile(list, filename):
    """Writing list of unique elements and number of each element.
    """
    f = open(filename, 'w')
    counter = Counter(list)
    for c in counter:
        f.write(c.encode('utf8'))
        f.write(' ' + str(counter[c]) + '\n')
    f.close()

def checkFeature(listFeature):
    """Writing file parse tree path, phrase type and subcategory features 
    """
    paths = []
    phraseTypes = []
    subCategories = []
    for record in listFeature:
        paths.append(record[1])
        phraseTypes.append(record[2])
        subCategories.append(record[6])
    writeFile(paths, 'path.txt')
    writeFile(phraseTypes, 'phrasetype.txt')
    writeFile(subCategories, 'subcat.txt')

def convertToDataFrame(listFeature):
    """Add colums' name of pandas database
    """
    listFeaturePD = pd.DataFrame(listFeature)
    listFeaturePD.columns = ['predicate', 'path tree', 'phrase type', 'position', 'voice', 'head word', 'sub category', 'path tree reduce', 'distance', 'head word type', 'function tag', 'predicate type']
    return listFeaturePD

def labelEncoderData(listFeature, listLE):
    """Return list of feature after encoding
    """
    listFeature = np.transpose(listFeature)
    for i in range(len(listFeature)):
        listFeature[i] = listLE[i].transform(listFeature[i])
    listFeature = np.transpose(listFeature)
    return listFeature

def getLabelEncoderParameter(listFeature, listLabel):
    listLE = []
    listFeature = np.asarray(listFeature)
    listFeature = np.transpose(listFeature)
    le = LabelEncoder()
    for i in range(len(listFeature)):
        le = LabelEncoder()
        le.fit(listFeature[i])
        listLE.append(le)
    listFeature = np.transpose(listFeature)
    leLabel = le = LabelEncoder()
    leLabel.fit(listLabel)
    listEncode = list(leLabel.classes_)
    return listLE, leLabel, listEncode

def getDataForIdenTrain(listFeature, listFeatureIdenName, listLabel, listEncode):
    """Return list of labels and corresponding list of features
    before Identification step
    """
    listFeatureIdenTrain = listFeature.loc[:,listFeatureIdenName]
    listFeatureIdenTrain = np.asarray(listFeatureIdenTrain)
    listLabelIdenTrain = []
    noneEncode = listEncode.index('None')
    for item in listLabel:
        if item == noneEncode:
            listLabelIdenTrain.append(0)
        else:
            listLabelIdenTrain.append(1)
    listLabelIdenTrain = np.asarray(listLabelIdenTrain)
    return listFeatureIdenTrain, listLabelIdenTrain

def getDataForIdenTest(listFeature, listFeatureIdenName, listLabel, listEncode):
    """Return list of labels and corresponding list of features
    before testing result of Identification step
    """
    listFeatureIdenTest = listFeature.loc[:,listFeatureIdenName]
    listFeatureIdenTest = np.asarray(listFeatureIdenTest)
    listLabelIdenTest = []
    noneEncode = listEncode.index('None')
    for item in listLabel:
        if item == noneEncode:
            listLabelIdenTest.append(0)
        else:
            listLabelIdenTest.append(1)
    listLabelIdenTest = np.asarray(listLabelIdenTest)	
    return listFeatureIdenTest, listLabelIdenTest

def getDataForClassTrain(listFeature, listFeatureClassName, listLabel, listEncode):
    """Return list of labels and corresponding list of features
    before Classification step
    """
    listFeatureClassTrain = listFeature.loc[:,listFeatureClassName]
    listFeatureClassTrain = listFeatureClassTrain.as_matrix()
    listFeatureClassTrain = listFeatureClassTrain.tolist()
    listFeatureClassTrainNew = []
    listLabelClassTrain = []
    noneEncode = listEncode.index('None')
    for i in range(len(listLabel)):
        if(listLabel[i] != noneEncode):
            listFeatureClassTrainNew.append(listFeatureClassTrain[i])
            listLabelClassTrain.append(listLabel[i])
    listFeatureClassTrainNew = np.asarray(listFeatureClassTrainNew)
    listLabelClassTrain = np.asarray(listLabelClassTrain)
    return listFeatureClassTrainNew, listLabelClassTrain

def getDataForClassTest(listFeature, listFeatureClassName, listEncode):
    """Return list of labels and corresponding list of features
    before testing classification step
    """
    listFeatureClassTest = listFeature.loc[:,listFeatureClassName]
    listFeatureClassTest = np.asarray(listFeatureClassTest)
    return listFeatureClassTest

def classificationNB(listFeature, listLabel, listFeatureTest):
    """Return list of predicted labels after applying Naive Bayes classifier
    """
    listFeatureNB = listFeature
    listFeatureNBTest = listFeatureTest
    clf = MultinomialNB()
    clf.fit(listFeatureNB, listLabel)
    listLabelPredict = clf.predict(listFeatureNBTest)
    return listLabelPredict

def classificationMaxEnt(listFeature, listLabel, listFeatureTest):
    """Return list of predicted labels after applying Maximum Entropy classifier
    """
    listFeatureMaxEnt = listFeature
    listFeatureMaxEntTest = listFeatureTest
    clf = LogisticRegression()
    clf.fit(listFeatureMaxEnt, listLabel)
    listLabelPredict = clf.predict(listFeatureMaxEntTest)
    return listLabelPredict

def classificationSVM(listFeature, listLabel, listFeatureTest):
    """Return list of predicted labels after applying Support Vector Machine classifier
    """
    listTotalFeature = np.concatenate((listFeature, listFeatureTest), axis=0)
    enc = OneHotEncoder()
    enc.fit(listTotalFeature)
    temp = enc.get_params(deep=True)
    listFeatureSVM = enc.transform(listFeature)
    listFeatureSVMTest = enc.transform(listFeatureTest)	
    clf = svm.LinearSVC(C=0.1)
    clf.fit(listFeatureSVM, listLabel)
    listLabelPredict = clf.predict(listFeatureSVMTest)
    densityMatrix = clf.decision_function(listFeatureSVMTest)
    return listLabelPredict, densityMatrix

def algorithmNB(listFeatureIdenTrain, listLabelIdenTrain, listFeatureIdenTest, listFeatureClassTrain, listLabelClassTrain, listFeatureClassTest, listLabelTest, listEncode, listLabelIdenTest):
    """Return result after applying Naive Bayes classifier
    """
    listFeatureClassTestAfterIden = []
    listFeatureClassTestTrue = []
    listLabelPredict = []
    listLabelPredictByClassTrue = []
    noneEncode = listEncode.index('None')
    listLabelPredictByIden = classificationNB(listFeatureIdenTrain, listLabelIdenTrain, listFeatureIdenTest)	
    for i in range(len(listFeatureIdenTest)):
        if(listLabelPredictByIden[i]==1):
            listFeatureClassTestAfterIden.append(listFeatureClassTest[i])
        if(listLabelIdenTest[i] == 1):
            listFeatureClassTestTrue.append(listFeatureClassTest[i])
    listLabelPredictByClassAfterIden = classificationNB(listFeatureClassTrain, listLabelClassTrain, listFeatureClassTestAfterIden)
    listLabelPredictByClassAfterIden = listLabelPredictByClassAfterIden.tolist()
    listLabelPredictByClass = classificationNB(listFeatureClassTrain, listLabelClassTrain, listFeatureClassTestTrue)
    listLabelPredictByClass = listLabelPredictByClass.tolist()	
    for i in range(len(listFeatureIdenTest)):
        if(listLabelPredictByIden[i]==0):
            listLabelPredict.append(noneEncode)
        else:
            listLabelPredict.append(listLabelPredictByClassAfterIden.pop(0))
        if(listLabelIdenTest[i]==0):
            listLabelPredictByClassTrue.append(noneEncode)
        else:
            listLabelPredictByClassTrue.append(listLabelPredictByClass.pop(0))		
    listLabelPredict = np.asarray(listLabelPredict)
    listLabelPredictByClassTrue = np.asarray(listLabelPredictByClassTrue)
    return listLabelPredict, listLabelPredictByIden, listLabelPredictByClassTrue

def algorithmMaxEnt(listFeatureIdenTrain, listLabelIdenTrain, listFeatureIdenTest, listFeatureClassTrain, listLabelClassTrain, listFeatureClassTest, listLabelTest, listEncode, listLabelIdenTest):
    """Return result after applying Maximum Entropy classifier
    """
    listFeatureClassTestAfterIden = []
    listFeatureClassTestTrue = []
    listLabelPredict = []
    listLabelPredictByClassTrue = []
    noneEncode = listEncode.index('None')
    listLabelPredictByIden = classificationMaxEnt(listFeatureIdenTrain, listLabelIdenTrain, listFeatureIdenTest)	
    for i in range(len(listFeatureIdenTest)):
        if(listLabelPredictByIden[i]==1):
            listFeatureClassTestAfterIden.append(listFeatureClassTest[i])
        if(listLabelIdenTest[i] == 1):
            listFeatureClassTestTrue.append(listFeatureClassTest[i])
    listLabelPredictByClassAfterIden = classificationMaxEnt(listFeatureClassTrain, listLabelClassTrain, listFeatureClassTestAfterIden)
    listLabelPredictByClassAfterIden = listLabelPredictByClassAfterIden.tolist()
    listLabelPredictByClass = classificationMaxEnt(listFeatureClassTrain, listLabelClassTrain, listFeatureClassTestTrue)
    listLabelPredictByClass = listLabelPredictByClass.tolist()	
    for i in range(len(listFeatureIdenTest)):
        if(listLabelPredictByIden[i]==0):
            listLabelPredict.append(noneEncode)
        else:
            listLabelPredict.append(listLabelPredictByClassAfterIden.pop(0))
        if(listLabelIdenTest[i]==0):
            listLabelPredictByClassTrue.append(noneEncode)
        else:
            listLabelPredictByClassTrue.append(listLabelPredictByClass.pop(0))		
    listLabelPredict = np.asarray(listLabelPredict)
    listLabelPredictByClassTrue = np.asarray(listLabelPredictByClassTrue)
    return listLabelPredict, listLabelPredictByIden, listLabelPredictByClassTrue

def algorithmSVM(listFeatureIdenTrain, listLabelIdenTrain, listFeatureIdenTest, listFeatureClassTrain, listLabelClassTrain, listFeatureClassTest, listLabelTest, listEncode, listLabelIdenTest):
    """Return result after applying Support Vector Machine classifier
    """
    listFeatureClassTestAfterIden = []
    listFeatureClassTestTrue = []
    listLabelPredict = []
    listLabelPredictByClassTrue = []
    noneEncode = listEncode.index('None')
    listLabelPredictByIden = classificationSVM(listFeatureIdenTrain, listLabelIdenTrain, listFeatureIdenTest)	
    for i in range(len(listFeatureIdenTest)):
        if(listLabelPredictByIden[i]==1):
            listFeatureClassTestAfterIden.append(listFeatureClassTest[i])
        if(listLabelIdenTest[i] == 1):
            listFeatureClassTestTrue.append(listFeatureClassTest[i])
    listLabelPredictByClassAfterIden = classificationSVM(listFeatureClassTrain, listLabelClassTrain, listFeatureClassTestAfterIden)
    listLabelPredictByClassAfterIden = listLabelPredictByClassAfterIden.tolist()
    listLabelPredictByClass = classificationSVM(listFeatureClassTrain, listLabelClassTrain, listFeatureClassTestTrue)
    listLabelPredictByClass = listLabelPredictByClass.tolist()	
    for i in range(len(listFeatureIdenTest)):
        if(listLabelPredictByIden[i]==0):
            listLabelPredict.append(noneEncode)
        else:
            listLabelPredict.append(listLabelPredictByClassAfterIden.pop(0))
        if(listLabelIdenTest[i]==0):
            listLabelPredictByClassTrue.append(noneEncode)
        else:
            listLabelPredictByClassTrue.append(listLabelPredictByClass.pop(0))		
    listLabelPredict = np.asarray(listLabelPredict)
    listLabelPredictByClassTrue = np.asarray(listLabelPredictByClassTrue)
    return listLabelPredict, listLabelPredictByIden, listLabelPredictByClassTrue

def getF1ScoreTotal(listLabelTest, listLabelPredict, numArg, listEncode):
    """Return Precision, Recall, F1Score (F1 Measure) for 2-step strategy
    """
    count1 = 0
    count2 = 0
    count3 = numArg
    noneEncode = listEncode.index('None')
    for i in range(len(listLabelTest)):
        if(listLabelPredict[i] != noneEncode):
            count1 += 1
            if(listLabelTest[i] == listLabelPredict[i]):
                count2 += 1
    precision = count2/float(count1)
    recall = count2/float(count3)
    f1Score = 2*(precision * recall)/(precision + recall)
    return precision, recall, f1Score

def getF1ScoreChunking(listLabelTest, numArg, listEncode):
    """Return Precision, Recall, F1Score (F1 Measure) for Constituent Extraction
    """
    count1 = 0
    count2 = len(listLabelTest)
    count3 = numArg
    noneEncode = listEncode.index('None')
    for i in range(len(listLabelTest)):
        if(listLabelTest[i] != noneEncode):
            count1 += 1
    precision = count1/float(count2)
    recall = count1/float(count3)
    f1Score = 2*(precision * recall)/(precision + recall)
    return precision, recall, f1Score

def getF1ScoreClassify1Stage(listLabelTest, listLabelPredict, listEncode):
    """Return Precision, Recall, F1Score (F1 Measure) for 1-step strategy (Classification step)
    """
    count1 = 0
    count2 = 0
    count3 = 0
    noneEncode = listEncode.index('None')
    for i in range(len(listLabelTest)):
        if(listLabelPredict[i] != noneEncode):
            count1 += 1
            if(listLabelTest[i] == listLabelPredict[i]):
                count2 += 1
        if(listLabelTest[i] != noneEncode):
            count3 += 1
    precision = count2/float(count1)
    recall = count2/float(count3)
    f1Score = 2*(precision * recall)/(precision + recall)	
    return precision, recall, f1Score

def getF1ScoreIden(listLabelTest, listLabelPredict, listEncode):
    """Return Precision, Recall, F1Score (F1 Measure) for Identification step
    """
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(len(listLabelTest)):
        if(listLabelPredict[i] != 0):
            count1 += 1
            if(listLabelTest[i] == listLabelPredict[i]):
                count2 += 1
        if(listLabelTest[i] != 0):
            count3 += 1
    precision = count2/float(count1)
    recall = count2/float(count3)
    f1Score = 2*(precision * recall)/(precision + recall)
    return precision, recall, f1Score

def getF1ScoreClass(listLabelTest, listLabelPredict, listEncode):
    """Return Precision, Recall, F1Score (F1 Measure) for Classification step
    """
    count1 = 0
    count2 = 0
    count3 = 0
    noneEncode = listEncode.index('None')
    for i in range(len(listLabelTest)):
        if(listLabelPredict[i] != noneEncode):
            count1 += 1
            if(listLabelTest[i] == listLabelPredict[i]):
                count2 += 1
        if(listLabelTest[i] != noneEncode):
            count3 += 1
    precision = count2/float(count1)
    recall = count2/float(count3)
    f1Score = 2*(precision * recall)/(precision + recall)	
    return precision, recall, f1Score

def evaluationNB2Stage(listFeatureTrain, listLabelTrain, listFeatureTest, listLabelTest, listFeatureIdenName, listFeatureClassName, listEncode, numArg):
    """Return result of Naive Bayes classifier 2-step strategy
    """
    listFeatureTrain = convertToDataFrame(listFeatureTrain)
    listFeatureTest = convertToDataFrame(listFeatureTest)	
    listFeatureIdenTrain, listLabelIdenTrain = getDataForIdenTrain(listFeatureTrain, listFeatureIdenName, listLabelTrain, listEncode)
    listFeatureClassTrain, listLabelClassTrain = getDataForClassTrain(listFeatureTrain, listFeatureClassName, listLabelTrain, listEncode)
    listFeatureIdenTest, listLabelIdenTest = getDataForIdenTest(listFeatureTest, listFeatureIdenName, listLabelTest, listEncode)
    listFeatureClassTest = getDataForClassTest(listFeatureTest, listFeatureClassName, listEncode)
    listLabelPredict, listLabelPredictByIden, listLabelPredictByClassTrue = algorithmNB(listFeatureIdenTrain, listLabelIdenTrain, listFeatureIdenTest, listFeatureClassTrain, listLabelClassTrain, listFeatureClassTest, listLabelTest, listEncode, listLabelIdenTest)
    precisionChunking, recallChunking, f1ScoreChunking = getF1ScoreChunking(listLabelTest, numArg, listEncode)
    precisionIden, recallIden, f1ScoreIden = getF1ScoreIden(listLabelIdenTest, listLabelPredictByIden, numArg)
    precisionClass, recallClass, f1ScoreClass = getF1ScoreClass(listLabelTest, listLabelPredict, listEncode)
    precision2Stage, recall2Stage, f1Score2Stage = getF1ScoreTotal(listLabelTest, listLabelPredict, numArg, listEncode)
    return precisionChunking, recallChunking, f1ScoreChunking, precisionIden, recallIden, f1ScoreIden, precisionClass, recallClass, f1ScoreClass,precision2Stage, recall2Stage, f1Score2Stage

def evaluationNB1Stage(listFeatureTrain, listLabelTrain, listFeatureTest, listLabelTest, numArg, listFeatureName, listEncode):
    """Return result of Naive Bayes classifier 1-step strategy
    """
    listFeatureTrain = convertToDataFrame(listFeatureTrain)
    listFeatureTest = convertToDataFrame(listFeatureTest)
    listFeatureTrain = listFeatureTrain.loc[:,listFeatureName]
    listFeatureTrain = np.asarray(listFeatureTrain)	
    listFeatureTest = listFeatureTest.loc[:,listFeatureName]
    listFeatureTest = np.asarray(listFeatureTest)
    listLabelTrain = np.asarray(listLabelTrain)
    listLabelTest = np.asarray(listLabelTest)
    listLabelPredict = classificationNB(listFeatureTrain, listLabelTrain, listFeatureTest)
    precisionChunking, recallChunking, f1ScoreChunking = getF1ScoreChunking(listLabelTest, numArg, listEncode)
    precisionClassify1Stage, recallClassify1Stage, f1ScoreClassify1Stage = getF1ScoreClassify1Stage(listLabelTest, listLabelPredict, listEncode)
    precision1Stage, recall1Stage, f1Score1Stage = getF1ScoreTotal(listLabelTest, listLabelPredict, numArg, listEncode)
    return precisionChunking, recallChunking, f1ScoreChunking, precision1Stage, precisionClassify1Stage, recallClassify1Stage, f1ScoreClassify1Stage, recall1Stage, f1Score1Stage

def evaluationMaxEnt2Stage(listFeatureTrain, listLabelTrain, listFeatureTest, listLabelTest, listFeatureIdenName, listFeatureClassName, listEncode, numArg):
    """Return result of Maximum Entropy classifier 2-step strategy
    """
    listFeatureTrain = convertToDataFrame(listFeatureTrain)
    listFeatureTest = convertToDataFrame(listFeatureTest)	
    listFeatureIdenTrain, listLabelIdenTrain = getDataForIdenTrain(listFeatureTrain, listFeatureIdenName, listLabelTrain, listEncode)
    listFeatureClassTrain, listLabelClassTrain = getDataForClassTrain(listFeatureTrain, listFeatureClassName, listLabelTrain, listEncode)
    listFeatureIdenTest, listLabelIdenTest = getDataForIdenTest(listFeatureTest, listFeatureIdenName, listLabelTest, listEncode)
    listFeatureClassTest = getDataForClassTest(listFeatureTest, listFeatureClassName, listEncode)
    listLabelPredict, listLabelPredictByIden, listLabelPredictByClassTrue = algorithmMaxEnt(listFeatureIdenTrain, listLabelIdenTrain, listFeatureIdenTest, listFeatureClassTrain, listLabelClassTrain, listFeatureClassTest, listLabelTest, listEncode, listLabelIdenTest)
    precisionChunking, recallChunking, f1ScoreChunking = getF1ScoreChunking(listLabelTest, numArg, listEncode)
    precisionIden, recallIden, f1ScoreIden = getF1ScoreIden(listLabelIdenTest, listLabelPredictByIden, numArg)
    precisionClass, recallClass, f1ScoreClass = getF1ScoreClass(listLabelTest, listLabelPredict, listEncode)
    precision2Stage, recall2Stage, f1Score2Stage = getF1ScoreTotal(listLabelTest, listLabelPredict, numArg, listEncode)
    return precisionChunking, recallChunking, f1ScoreChunking, precisionIden, recallIden, f1ScoreIden, precisionClass, recallClass, f1ScoreClass,precision2Stage, recall2Stage, f1Score2Stage

def evaluationMaxEnt1Stage(listFeatureTrain, listLabelTrain, listFeatureTest, listLabelTest, numArg, listFeatureName, listEncode):
    """Return result of Maximum Entropy classifier 1-step strategy
    """
    listFeatureTrain = convertToDataFrame(listFeatureTrain)
    listFeatureTest = convertToDataFrame(listFeatureTest)
    listFeatureTrain = listFeatureTrain.loc[:,listFeatureName]
    listFeatureTrain = np.asarray(listFeatureTrain)	
    listFeatureTest = listFeatureTest.loc[:,listFeatureName]
    listFeatureTest = np.asarray(listFeatureTest)
    listLabelTrain = np.asarray(listLabelTrain)
    listLabelTest = np.asarray(listLabelTest)
    listLabelPredict = classificationMaxEnt(listFeatureTrain, listLabelTrain, listFeatureTest)
    precisionChunking, recallChunking, f1ScoreChunking = getF1ScoreChunking(listLabelTest, numArg, listEncode)
    precisionClassify1Stage, recallClassify1Stage, f1ScoreClassify1Stage = getF1ScoreClassify1Stage(listLabelTest, listLabelPredict, listEncode)
    precision1Stage, recall1Stage, f1Score1Stage = getF1ScoreTotal(listLabelTest, listLabelPredict, numArg, listEncode)
    return precisionChunking, recallChunking, f1ScoreChunking, precision1Stage, precisionClassify1Stage, recallClassify1Stage, f1ScoreClassify1Stage, recall1Stage, f1Score1Stage

def evaluationSVM2Stage(listFeatureTrain, listLabelTrain, listFeatureTest, listLabelTest, listFeatureIdenName, listFeatureClassName, listEncode, numArg):
    """Return result of Support Vector Machine classifier 2-step strategy
    """
    listFeatureTrain = convertToDataFrame(listFeatureTrain)
    listFeatureTest = convertToDataFrame(listFeatureTest)	
    listFeatureIdenTrain, listLabelIdenTrain = getDataForIdenTrain(listFeatureTrain, listFeatureIdenName, listLabelTrain, listEncode)
    listFeatureClassTrain, listLabelClassTrain = getDataForClassTrain(listFeatureTrain, listFeatureClassName, listLabelTrain, listEncode)
    listFeatureIdenTest, listLabelIdenTest = getDataForIdenTest(listFeatureTest, listFeatureIdenName, listLabelTest, listEncode)
    listFeatureClassTest = getDataForClassTest(listFeatureTest, listFeatureClassName, listEncode)
    listLabelPredict, listLabelPredictByIden, listLabelPredictByClassTrue = algorithmSVM(listFeatureIdenTrain, listLabelIdenTrain, listFeatureIdenTest, listFeatureClassTrain, listLabelClassTrain, listFeatureClassTest, listLabelTest, listEncode, listLabelIdenTest)
    precisionChunking, recallChunking, f1ScoreChunking = getF1ScoreChunking(listLabelTest, numArg, listEncode)
    precisionIden, recallIden, f1ScoreIden = getF1ScoreIden(listLabelIdenTest, listLabelPredictByIden, numArg)
    precisionClass, recallClass, f1ScoreClass = getF1ScoreClass(listLabelTest, listLabelPredict, listEncode)
    precision2Stage, recall2Stage, f1Score2Stage = getF1ScoreTotal(listLabelTest, listLabelPredict, numArg, listEncode)
    return precisionChunking, recallChunking, f1ScoreChunking, precisionIden, recallIden, f1ScoreIden, precisionClass, recallClass, f1ScoreClass,precision2Stage, recall2Stage, f1Score2Stage

def evaluationSVM1Stage(listFeatureTrain, listLabelTrain, listFeatureTest, listLabelTest, numArg, listFeatureName, listEncode, listNumArgPerSen, listPredicateType):
    """Return result of Support Vector Machine classifier 1-step strategy
    """
    listFeatureTrain = convertToDataFrame(listFeatureTrain)
    listFeatureTest = convertToDataFrame(listFeatureTest)
    listFeatureTrain = listFeatureTrain.loc[:,listFeatureName]
    listFeatureTrain = np.asarray(listFeatureTrain)	
    listFeatureTest = listFeatureTest.loc[:,listFeatureName]
    listFeatureTest = np.asarray(listFeatureTest)
    listLabelTrain = np.asarray(listLabelTrain)
    listLabelTest = np.asarray(listLabelTest)
    listLabelPredict, densityMatrix = classificationSVM(listFeatureTrain, listLabelTrain, listFeatureTest)
    count = 0
    listLabelILP = []
    listVariable = []
    listPredicateType = np.asarray(listPredicateType)
    for item in listNumArgPerSen:
        predicateType = listPredicateType[count]
        tempMatrix = densityMatrix[count:(count+item), :]
        listLabelTemp, listVariableTemp = ilpSolving(tempMatrix, predicateType)
        listLabelILP.append(listLabelTemp)
        listVariable.append(listVariableTemp)
        count += item
    listLabelILPNew = [item for sublist in listLabelILP for item in sublist]
    #print numArg
    #print sum(listNumArgPerSen)
    #print len(listLabelPredict)
    #print len(listLabelILPNew)
    precisionArg, recallArg, f1ScoreArg = evaluateArg(listLabelTest, listLabelILPNew)
    precisionChunking, recallChunking, f1ScoreChunking = getF1ScoreChunking(listLabelTest, numArg, listEncode)
    precisionClassify1Stage, recallClassify1Stage, f1ScoreClassify1Stage = getF1ScoreClassify1Stage(listLabelTest, listLabelILPNew, listEncode)
    precision1Stage, recall1Stage, f1Score1Stage = getF1ScoreTotal(listLabelTest, listLabelILPNew, numArg, listEncode)
    return precisionChunking, recallChunking, f1ScoreChunking, precision1Stage, precisionClassify1Stage, recallClassify1Stage, f1ScoreClassify1Stage, recall1Stage, f1Score1Stage, listLabelILPNew, listLabelPredict, densityMatrix, listVariable, precisionArg, recallArg, f1ScoreArg

def crossValidation(groupListLabel, groupListFeature, listLE, leLabel, foldNumber, groupInfo):
    """k-fold Cross validation
    """
    listOfListFeatureTrain = []
    listOfListFeatureTest = []
    listOfListLabelTrain = []
    listOfListLabelTest = []
    listOfListNumArg = []
    listOfListPredicateType = []
    for i in range(foldNumber):
        listFeatureTest = groupListFeature[i]
        listPredicateType = (np.asarray(listFeatureTest))[:, 11]
        listLabelTest = groupListLabel[i]
        listNumArg = groupInfo[i]
        numArg = 0
        for k in range(len(listNumArg)):
            numArg += int(listNumArg[k][1])
        if (i == 0):
            listFeatureTrain = groupListFeature[1]
            listLabelTrain = groupListLabel[1]
            init = 1
        else:
            listFeatureTrain = groupListFeature[0]
            listLabelTrain = groupListLabel[0]
            init = 0
        for j in range(foldNumber):
            if((j != i) and (j != init)):
                listFeatureTrain = np.concatenate((listFeatureTrain, groupListFeature[j]), axis=0)
                listLabelTrain = np.concatenate((listLabelTrain, groupListLabel[j]), axis=0)

        listFeatureTrain = labelEncoderData(listFeatureTrain, listLE)
        listFeatureTrain = listFeatureTrain.astype(int)
        listLabelTrain = leLabel.transform(listLabelTrain)
        listFeatureTest = labelEncoderData(listFeatureTest, listLE)
        listFeatureTest = listFeatureTest.astype(int)
        listLabelTest = leLabel.transform(listLabelTest)
        listOfListFeatureTrain.append(listFeatureTrain)
        listOfListFeatureTest.append(listFeatureTest)
        listOfListLabelTrain.append(listLabelTrain)
        listOfListLabelTest.append(listLabelTest)
        listOfListNumArg.append(numArg)
        listOfListPredicateType.append(listPredicateType)
    return listOfListFeatureTrain, listOfListFeatureTest, listOfListLabelTrain, listOfListLabelTest, listOfListNumArg, listOfListPredicateType

def crossValidationNB2Stage(listOfListFeatureTrain, listOfListFeatureTest, listOfListLabelTrain, listOfListLabelTest, listFeatureIdenName, listFeatureClassName, foldNumber, listOfListNumArg, listEncode):
    """k-fold Cross validation for Naive Bayes 2-step strategy
    """
    listPrecisionChunking = []
    listRecallChunking = []
    listF1ScoreChunking = []		
    listPrecisionIden = []
    listRecallIden = []
    listF1ScoreIden = []
    listPrecisionClass = []
    listRecallClass = []
    listF1ScoreClass = []
    listPrecision2Stage = []
    listRecall2Stage = []
    listF1Score2Stage = []	
    for i in range(foldNumber):
        listFeatureTrain = listOfListFeatureTrain[i]
        listLabelTrain = listOfListLabelTrain[i]
        listFeatureTest = listOfListFeatureTest[i]
        listLabelTest = listOfListLabelTest[i]
        numArg = listOfListNumArg[i]
        precisionChunking, recallChunking, f1ScoreChunking, precisionIden, recallIden, f1ScoreIden, precisionClass, recallClass, f1ScoreClass,precision2Stage, recall2Stage, f1Score2Stage = evaluationNB2Stage(listFeatureTrain, listLabelTrain, listFeatureTest, listLabelTest, listFeatureIdenName, listFeatureClassName, listEncode, numArg)
        listPrecisionChunking.append(precisionChunking)
        listRecallChunking.append(recallChunking)
        listF1ScoreChunking.append(f1ScoreChunking)			
        listPrecisionIden.append(precisionIden)
        listRecallIden.append(recallIden)
        listF1ScoreIden.append(f1ScoreIden)
        listPrecisionClass.append(precisionClass)
        listRecallClass.append(recallClass)
        listF1ScoreClass.append(f1ScoreClass)
        listPrecision2Stage.append(precision2Stage)
        listRecall2Stage.append(recall2Stage)
        listF1Score2Stage.append(f1Score2Stage)		
    return listPrecisionChunking, listRecallChunking, listF1ScoreChunking, listPrecisionIden, listRecallIden, listF1ScoreIden, listPrecisionClass, listRecallClass, listF1ScoreClass, listPrecision2Stage, listRecall2Stage, listF1Score2Stage

def crossValidationNB1Stage(listOfListFeatureTrain, listOfListFeatureTest, listOfListLabelTrain, listOfListLabelTest, listFeatureName, foldNumber, listOfListNumArg, listEncode):
    """k-fold Cross validation for Naive Bayes 1-step strategy
    """
    listPrecisionChunking = []
    listRecallChunking = []
    listF1ScoreChunking = []
    listPrecisionClassify1Stage = []
    listRecallClassify1Stage = []
    listF1ScoreClassify1Stage = []	
    listPrecision1Stage = []
    listRecall1Stage = []
    listF1Score1Stage = []	
    for i in range(foldNumber):
        listFeatureTrain = listOfListFeatureTrain[i]
        listLabelTrain = listOfListLabelTrain[i]
        listFeatureTest = listOfListFeatureTest[i]
        listLabelTest = listOfListLabelTest[i]
        numArg = listOfListNumArg[i]
        precisionChunking, recallChunking, f1ScoreChunking, precision1Stage, precisionClassify1Stage, recallClassify1Stage, f1ScoreClassify1Stage, recall1Stage, f1Score1Stage = evaluationNB1Stage(listFeatureTrain, listLabelTrain, listFeatureTest, listLabelTest, numArg, listFeatureName, listEncode)
        listPrecisionChunking.append(precisionChunking)
        listRecallChunking.append(recallChunking)
        listF1ScoreChunking.append(f1ScoreChunking)
        listPrecisionClassify1Stage.append(precisionClassify1Stage)
        listRecallClassify1Stage.append(recallClassify1Stage)
        listF1ScoreClassify1Stage.append(f1ScoreClassify1Stage)		
        listPrecision1Stage.append(precision1Stage)
        listRecall1Stage.append(recall1Stage)
        listF1Score1Stage.append(f1Score1Stage)
    return listPrecisionChunking, listRecallChunking, listF1ScoreChunking, listPrecisionClassify1Stage, listRecallClassify1Stage, listF1ScoreClassify1Stage, listPrecision1Stage, listRecall1Stage, listF1Score1Stage

def crossValidationMaxEnt2Stage(listOfListFeatureTrain, listOfListFeatureTest, listOfListLabelTrain, listOfListLabelTest, listFeatureIdenName, listFeatureClassName, foldNumber, listOfListNumArg, listEncode):
    """k-fold Cross validation for Maximum Entropy classifier 2-step strategy
    """
    listPrecisionChunking = []
    listRecallChunking = []
    listF1ScoreChunking = []		
    listPrecisionIden = []
    listRecallIden = []
    listF1ScoreIden = []
    listPrecisionClass = []
    listRecallClass = []
    listF1ScoreClass = []
    listPrecision2Stage = []
    listRecall2Stage = []
    listF1Score2Stage = []	
    for i in range(foldNumber):
        listFeatureTrain = listOfListFeatureTrain[i]
        listLabelTrain = listOfListLabelTrain[i]
        listFeatureTest = listOfListFeatureTest[i]
        listLabelTest = listOfListLabelTest[i]
        numArg = listOfListNumArg[i]
        precisionChunking, recallChunking, f1ScoreChunking, precisionIden, recallIden, f1ScoreIden, precisionClass, recallClass, f1ScoreClass,precision2Stage, recall2Stage, f1Score2Stage = evaluationMaxEnt2Stage(listFeatureTrain, listLabelTrain, listFeatureTest, listLabelTest, listFeatureIdenName, listFeatureClassName, listEncode, numArg)
        listPrecisionChunking.append(precisionChunking)
        listRecallChunking.append(recallChunking)
        listF1ScoreChunking.append(f1ScoreChunking)			
        listPrecisionIden.append(precisionIden)
        listRecallIden.append(recallIden)
        listF1ScoreIden.append(f1ScoreIden)
        listPrecisionClass.append(precisionClass)
        listRecallClass.append(recallClass)
        listF1ScoreClass.append(f1ScoreClass)
        listPrecision2Stage.append(precision2Stage)
        listRecall2Stage.append(recall2Stage)
        listF1Score2Stage.append(f1Score2Stage)		
    return listPrecisionChunking, listRecallChunking, listF1ScoreChunking, listPrecisionIden, listRecallIden, listF1ScoreIden, listPrecisionClass, listRecallClass, listF1ScoreClass, listPrecision2Stage, listRecall2Stage, listF1Score2Stage

def crossValidationMaxEnt1Stage(listOfListFeatureTrain, listOfListFeatureTest, listOfListLabelTrain, listOfListLabelTest, listFeatureName, foldNumber, listOfListNumArg, listEncode):
    """k-fold Cross validation for Maximum Entropy classifier 1-step strategy
    """
    listPrecisionChunking = []
    listRecallChunking = []
    listF1ScoreChunking = []
    listPrecisionClassify1Stage = []
    listRecallClassify1Stage = []
    listF1ScoreClassify1Stage = []	
    listPrecision1Stage = []
    listRecall1Stage = []
    listF1Score1Stage = []	
    for i in range(foldNumber):
        listFeatureTrain = listOfListFeatureTrain[i]
        listLabelTrain = listOfListLabelTrain[i]
        listFeatureTest = listOfListFeatureTest[i]
        listLabelTest = listOfListLabelTest[i]
        numArg = listOfListNumArg[i]
        precisionChunking, recallChunking, f1ScoreChunking, precision1Stage, precisionClassify1Stage, recallClassify1Stage, f1ScoreClassify1Stage, recall1Stage, f1Score1Stage = evaluationMaxEnt1Stage(listFeatureTrain, listLabelTrain, listFeatureTest, listLabelTest, numArg, listFeatureName, listEncode)
        listPrecisionChunking.append(precisionChunking)
        listRecallChunking.append(recallChunking)
        listF1ScoreChunking.append(f1ScoreChunking)
        listPrecisionClassify1Stage.append(precisionClassify1Stage)
        listRecallClassify1Stage.append(recallClassify1Stage)
        listF1ScoreClassify1Stage.append(f1ScoreClassify1Stage)		
        listPrecision1Stage.append(precision1Stage)
        listRecall1Stage.append(recall1Stage)
        listF1Score1Stage.append(f1Score1Stage)
    return listPrecisionChunking, listRecallChunking, listF1ScoreChunking, listPrecisionClassify1Stage, listRecallClassify1Stage, listF1ScoreClassify1Stage, listPrecision1Stage, listRecall1Stage, listF1Score1Stage

def crossValidationSVM2Stage(listOfListFeatureTrain, listOfListFeatureTest, listOfListLabelTrain, listOfListLabelTest, listFeatureIdenName, listFeatureClassName, foldNumber, listOfListNumArg, listEncode):
    """k-fold Cross validation for Support Vector Machine classifier 2-step strategy
    """
    listPrecisionChunking = []
    listRecallChunking = []
    listF1ScoreChunking = []		
    listPrecisionIden = []
    listRecallIden = []
    listF1ScoreIden = []
    listPrecisionClass = []
    listRecallClass = []
    listF1ScoreClass = []
    listPrecision2Stage = []
    listRecall2Stage = []
    listF1Score2Stage = []	
    for i in range(foldNumber):
        listFeatureTrain = listOfListFeatureTrain[i]
        listLabelTrain = listOfListLabelTrain[i]
        listFeatureTest = listOfListFeatureTest[i]
        listLabelTest = listOfListLabelTest[i]
        numArg = listOfListNumArg[i]
        precisionChunking, recallChunking, f1ScoreChunking, precisionIden, recallIden, f1ScoreIden, precisionClass, recallClass, f1ScoreClass,precision2Stage, recall2Stage, f1Score2Stage = evaluationSVM2Stage(listFeatureTrain, listLabelTrain, listFeatureTest, listLabelTest, listFeatureIdenName, listFeatureClassName, listEncode, numArg)
        listPrecisionChunking.append(precisionChunking)
        listRecallChunking.append(recallChunking)
        listF1ScoreChunking.append(f1ScoreChunking)			
        listPrecisionIden.append(precisionIden)
        listRecallIden.append(recallIden)
        listF1ScoreIden.append(f1ScoreIden)
        listPrecisionClass.append(precisionClass)
        listRecallClass.append(recallClass)
        listF1ScoreClass.append(f1ScoreClass)
        listPrecision2Stage.append(precision2Stage)
        listRecall2Stage.append(recall2Stage)
        listF1Score2Stage.append(f1Score2Stage)		
    return listPrecisionChunking, listRecallChunking, listF1ScoreChunking, listPrecisionIden, listRecallIden, listF1ScoreIden, listPrecisionClass, listRecallClass, listF1ScoreClass, listPrecision2Stage, listRecall2Stage, listF1Score2Stage

def crossValidationSVM1Stage(listOfListFeatureTrain, listOfListFeatureTest, listOfListLabelTrain, listOfListLabelTest, listFeatureName, foldNumber, listOfListNumArg, listEncode, listOfListNumArgPerSen, listOfListPredicateType):
    """k-fold Cross validation for Support Vector Machine classifier 1-step strategy
    """
    listPrecisionChunking = []
    listRecallChunking = []
    listF1ScoreChunking = []
    listPrecisionClassify1Stage = []
    listRecallClassify1Stage = []
    listF1ScoreClassify1Stage = []	
    listPrecision1Stage = []
    listRecall1Stage = []
    listF1Score1Stage = []
    listOfListLabelPredict = []
    listOfListLabelILP = []
    listDensityMatrix = []
    listOfListVariable = []
    listPrecisionArg = []
    listRecallArg = []
    listF1ScoreArg = []
    for i in range(foldNumber):
        print 'Running Fold ' + str(i)
        listFeatureTrain = listOfListFeatureTrain[i]
        listLabelTrain = listOfListLabelTrain[i]
        listFeatureTest = listOfListFeatureTest[i]
        listLabelTest = listOfListLabelTest[i]
        listPredicateType = listOfListPredicateType[i]
        numArg = listOfListNumArg[i]
        listNumArgPerSen = listOfListNumArgPerSen[i]
        precisionChunking, recallChunking, f1ScoreChunking, precision1Stage, precisionClassify1Stage, recallClassify1Stage, f1ScoreClassify1Stage, recall1Stage, f1Score1Stage, listLabelILPNew, listLabelPredict, densityMatrix, listVariable, precisionArg, recallArg, f1ScoreArg = evaluationSVM1Stage(listFeatureTrain, listLabelTrain, listFeatureTest, listLabelTest, numArg, listFeatureName, listEncode, listNumArgPerSen, listPredicateType)
        listPrecisionChunking.append(precisionChunking)
        listRecallChunking.append(recallChunking)
        listF1ScoreChunking.append(f1ScoreChunking)
        listPrecisionClassify1Stage.append(precisionClassify1Stage)
        listRecallClassify1Stage.append(recallClassify1Stage)
        listF1ScoreClassify1Stage.append(f1ScoreClassify1Stage)		
        listPrecision1Stage.append(precision1Stage)
        listRecall1Stage.append(recall1Stage)
        listF1Score1Stage.append(f1Score1Stage)
        listOfListLabelPredict.append(listLabelPredict)
        listOfListLabelILP.append(listLabelILPNew)
        listDensityMatrix.append(densityMatrix)
        listPrecisionArg.append(precisionArg)
        listRecallArg.append(recallArg)
        listF1ScoreArg.append(f1ScoreArg)
    return listPrecisionChunking, listRecallChunking, listF1ScoreChunking, listPrecisionClassify1Stage, listRecallClassify1Stage, listF1ScoreClassify1Stage, listPrecision1Stage, listRecall1Stage, listF1Score1Stage, listOfListLabelPredict, listOfListLabelILP, listDensityMatrix, listOfListVariable, listPrecisionArg, listRecallArg, listF1ScoreArg

def output1Stage(listPrecisionChunking, listRecallChunking, listF1ScoreChunking, listPrecisionClassify1Stage, listRecallClassify1Stage, listF1ScoreClassify1Stage, listPrecision1Stage, listRecall1Stage, listF1Score1Stage, listPrecisionArg, listRecallArg, listF1ScoreArg):
    """Print result of 1-step strategy
    """
    print "Evaluate Chunking: "
    s = 'Precision: ' + repr(sum(listPrecisionChunking)/len(listPrecisionChunking))
    print s
    s = 'Recall: ' + repr(sum(listRecallChunking)/len(listRecallChunking))
    print s
    s = 'F1Score: ' + repr(sum(listF1ScoreChunking)/len(listF1ScoreChunking))
    print s
    print "Evaluate Classify: "
    s = 'Precision: ' + repr(sum(listPrecisionClassify1Stage)/len(listPrecisionClassify1Stage))
    print s
    s = 'Recall: ' + repr(sum(listRecallClassify1Stage)/len(listRecallClassify1Stage))
    print s
    s = 'F1Score: ' + repr(sum(listF1ScoreClassify1Stage)/len(listF1ScoreClassify1Stage))
    print s
    print "Evaluate Classification Arguments: "
    s = 'Precision: ' + repr(sum(listPrecisionArg)/len(listPrecisionArg))
    print s
    s = 'Recall: ' + repr(sum(listRecallArg)/len(listRecallArg))
    print s
    s = 'F1Score: ' + repr(sum(listF1ScoreArg)/len(listF1ScoreArg))
    print s
    print "Evaluate Total: "
    s = 'Precision: ' + repr(sum(listPrecision1Stage)/len(listPrecision1Stage))
    print s
    s = 'Recall: ' + repr(sum(listRecall1Stage)/len(listRecall1Stage))
    print s
    s = 'F1Score: ' + repr(sum(listF1Score1Stage)/len(listF1Score1Stage))
    print s

def output2Stage(listPrecisionChunking, listRecallChunking, listF1ScoreChunking, listPrecisionIden, listRecallIden, listF1ScoreIden, listPrecisionClass, listRecallClass, listF1ScoreClass, listPrecision2Stage, listRecall2Stage, listF1Score2Stage):
    """Print result 2-step strategy
    """
    print "Evaluate Chunking: "
    s = 'Precision: ' + repr(sum(listPrecisionChunking)/len(listPrecisionChunking))
    print s
    s = 'Recall: ' + repr(sum(listRecallChunking)/len(listRecallChunking))
    print s
    s = 'F1Score: ' + repr(sum(listF1ScoreChunking)/len(listF1ScoreChunking))
    print s
    print "Evaluate Identification: "
    s = 'Precision: ' + repr(sum(listPrecisionIden)/len(listPrecisionIden))
    print s
    s = 'Recall: ' + repr(sum(listRecallIden)/len(listRecallIden))
    print s
    s = 'F1Score: ' + repr(sum(listF1ScoreIden)/len(listF1ScoreIden))
    print s
    print "Evaluate Classification: "
    s = 'Precision: ' + repr(sum(listPrecisionClass)/len(listPrecisionClass))
    print s
    s = 'Recall: ' + repr(sum(listRecallClass)/len(listRecallClass))
    print s
    s = 'F1Score: ' + repr(sum(listF1ScoreClass)/len(listF1ScoreClass))
    print s
    print "Evaluate Total: "
    s = 'Precision: ' + repr(sum(listPrecision2Stage)/len(listPrecision2Stage))
    print s
    s = 'Recall: ' + repr(sum(listRecall2Stage)/len(listRecall2Stage))
    print s
    s = 'F1Score: ' + repr(sum(listF1Score2Stage)/len(listF1Score2Stage))
    print s

def createDataFromText(filename):
    fr1 = open(filename)
    listWord = []
    listVec = []
    info = fr1.readline().split()
    numTrain = int(info[0])
    numFeature = int(info[1])
    for line in fr1.readlines():
        listFromLine = line.split()
        listWord.append(listFromLine[0])
        listVec.append(map(float, listFromLine[1:numFeature+1]))
    return listWord, listVec

def miniBatchKMeans(listVec, numCluster):
    estimator = MiniBatchKMeans(n_clusters=numCluster, init='k-means++', max_iter=1000, batch_size=500, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)
    estimator.fit(listVec)
    labels = estimator.labels_
    clusterCenters = estimator.cluster_centers_
    return labels, clusterCenters

def GMM(listVec, numCluster):
    estimator = mixture.GMM(n_components=numCluster, covariance_type='diag', random_state=None, thresh=0.01, min_covar=0.001, n_iter=500, n_init=1, params='wmc', init_params='wmc')
    estimator.fit(listVec)
    weights = estimator.weights_
    means = estimator.means_
    covars = estimator.covars_
    converged = estimator.converged_
    labels = estimator.predict(listVec1)
    return weights, means, covars, converged, labels

def getDataInformation(listA):
    tempList = []
    for args in listA:
        for arg in args:
            tempList.append(reformLabel(arg[0]))
    counter = Counter(tempList)
    uniqueList = []
    for c in counter:
        uniqueList.append([c, counter[c]])
    return uniqueList

# def kFold(listID, listTree, listRel, listArg, listWordName, listCluster, numberGroup, listLabelReduce):
#     """return k list of sentences for k-fold cross validation
#     """
#     groupListID = []
#     groupListTree = []
#     groupListRel = []
#     groupListArg = []
#     groupListLabel = []
#     groupListFeature = []
#     groupInfo = []
#     groupListNumArg = []
#     for i in range(numberGroup):
#         groupListID.append([])
#         groupListTree.append([])
#         groupListRel.append([])
#         groupListArg.append([])
#     for i in range(len(listTree)):
#         groupListID[(i%numberGroup)].append(listID[i])
#         groupListTree[(i%numberGroup)].append(listTree[i])
#         groupListRel[(i%numberGroup)].append(listRel[i])
#         groupListArg[(i%numberGroup)].append(listArg[i])
#     for i in range(numberGroup):
#         listNumArg = []
#         groupInfo.append(sorted(getDataInformation(groupListArg[i])))
#         listID1, listTree1, listRel1, listArg1 = chunking(groupListID[i], groupListTree[i], groupListRel[i], groupListArg[i])
#         listLabel, listFeature, listCount = getFeature(listID1, listTree1, listRel1, listArg1, listWordName, listCluster)
#         listLabel = getListLabelReduce(listLabel, listLabelReduce)
#         groupListLabel.append(listLabel)
#         groupListFeature.append(listFeature)
#         for j in range(len(groupListArg[i])):
#             listNumArg.append(len(groupListArg[i][j]))
#         groupListNumArg.append(listCount)
#     return groupInfo, groupListLabel, groupListFeature, groupListNumArg

## New Code
def kFold(listID, listTree, listRel, listArg, listWordName, listCluster, numberGroup, listLabelReduce):
    """return k list of sentences for k-fold cross validation
    """
    groupListID = []
    groupListTree = []
    groupListRel = []
    groupListArg = []
    groupListLabel = []
    groupListFeature = []
    groupInfo = []
    groupListNumArg = []
    for i in range(numberGroup):
        groupListID.append([])
        groupListTree.append([])
        groupListRel.append([])
        groupListArg.append([])
    for i in range(len(listTree)):
        groupListID[(i%numberGroup)].append(listID[i])
        groupListTree[(i%numberGroup)].append(listTree[i])
        groupListRel[(i%numberGroup)].append(listRel[i])
        groupListArg[(i%numberGroup)].append(listArg[i])
    for i in range(numberGroup):
        listNumArg = []
        groupInfo.append(sorted(getDataInformation(groupListArg[i])))
        # listID1, listTree1, listRel1, listArg1 = chunking(groupListID[i], groupListTree[i], groupListRel[i], groupListArg[i])
        print 'Group: ' + str(i)
        listLabel, listFeature, listCount = getFeatureAllNode(groupListID[i], groupListTree[i], groupListRel[i], groupListArg[i], listWordName, listCluster)
        listLabel = getListLabelReduce(listLabel, listLabelReduce)
        groupListLabel.append(listLabel)
        groupListFeature.append(listFeature)
        for j in range(len(groupListArg[i])):
            listNumArg.append(len(groupListArg[i][j]))
        groupListNumArg.append(listCount)
    return groupInfo, groupListLabel, groupListFeature, groupListNumArg

def getListLabelReduce(listLabel, listLabelReduce):
    for i in range(len(listLabel)):
        if(listLabel[i] not in listLabelReduce):
            listLabel[i] = 'None'
    return listLabel

def uniqueArg(listA):
    listB = []
    for arg in listA:
        listB.append(arg[0]+':'+arg[1])
    listB = unique(listB)
    listC = []
    for arg in listB:
        pos = arg.find(':')
        listC.append([arg[:pos], arg[pos+1:]])
    return listC

def inTree(tree, ss):
    """Return True if string ss is existed on "tree"
    		  False if not
    """
    if ss == '':
        return False
    if len(tree.search_nodes(word = ss)) > 0:
        return True
    else:
        leaves = tree.get_leaves()
        for i in range(len(leaves)):
            if ss.find(leaves[i].word) == 0:
                temp = ''
                temp += leaves[i].word
                for j in range(i+1, len(leaves)):
                    if ss.find(temp + ' ' + leaves[j].word) == 0:
                        temp += ' ' + leaves[j].word
                    else:
                        if ss == temp:
                            return True
                        else:
                            temp = ''
                if ss == temp:
                    return True
    return False

def extractFromMultiRel(listID, listTree, listRel, listArg):
    """Extracting the "simple sentence" in sentences which have more than 1 relation
    """
    newListID = []
    newListTree = []
    newListRel = []
    newListArg = []
    for i in range(len(listID)):
        if len(listRel[i]) > 1:
            listNode1 = []
            for node in listTree[i].get_descendants():
                if isSType(node.name):
                    listNode1.append(node)
            listNode2 = []
            for node in listNode1:
                found = False
                for nn in node.get_descendants():
                    if isSType(nn.name):
                        found = True
                        break
                if not found:
                    listNode2.append(node)
            for node in listNode2:
                sentence = ''
                for leaf in node:
                    sentence += leaf.word + ' '
                sentence = sentence.strip()
                cnt = 0
                newRel = ''
                for rel in listRel[i]:
                    cnt += sentence.count(rel)
                    if cnt == 1 and sentence.count(rel) == 1:
                        newRel = rel
                if cnt == 1 and inTree(node, newRel):
                    args = []
                    rels = []
                    newListID.append(listID[i])
                    newListTree.append(node.copy())
                    rels.append(newRel)
                    for arg in listArg[i]:
                        if (arg[1].find(newRel) == -1) and (sentence.find(arg[1]) != -1):
                            args.append(arg)
                    newListRel.append(rels)
                    newListArg.append(uniqueArg(args))
    return newListID, newListTree, newListRel, newListArg

def mergeData(listID1, listTree1, listRel1, listArg1, listID2, listTree2, listRel2, listArg2):
    """Merging sentences of 2 list.
    """
    totalListID = []
    totalListTree = []
    totalListRel = []
    totalListArg = []
    totalListID.extend(listID1)
    totalListID.extend(listID2)
    totalListTree.extend(listTree1)
    totalListTree.extend(listTree2)
    totalListRel.extend(listRel1)
    totalListRel.extend(listRel2)
    totalListArg.extend(listArg1)
    totalListArg.extend(listArg2)
    return totalListID, totalListTree, totalListRel, totalListArg

def crossValidationTotal(groupListLabel, groupListFeature, listLE, leLabel, foldNumber, groupInfo):
    listOfListFeatureTrain = []
    listOfListLabelTrain = []
    listOfListNumArg = []
    for i in range(foldNumber):
        listFeatureTrain = groupListFeature[i]
        listLabelTrain = groupListLabel[i]
        listNumArg = groupInfo[i]
        numArg = 0
        for k in range(len(listNumArg)):
            numArg += int(listNumArg[k][1])
        listFeatureTrain = labelEncoderData(listFeatureTrain, listLE)
        listFeatureTrain = listFeatureTrain.astype(int)
        listLabelTrain = leLabel.transform(listLabelTrain)
        listOfListFeatureTrain.append(listFeatureTrain)
        listOfListLabelTrain.append(listLabelTrain)
        listOfListNumArg.append(numArg)
    return listOfListFeatureTrain, listOfListLabelTrain, listOfListNumArg

def crossValidationSVMTotal(listOfListFeatureTrain, listOfListLabelTrain, listFeatureName, foldNumber, listOfListNumArg, listEncode):
    for i in range(foldNumber):
        listFeatureTrain = listOfListFeatureTrain[i]
        listLabelTrain = listOfListLabelTrain[i]
        numArg = listOfListNumArg[i]
        listFeatureTrain = convertToDataFrame(listFeatureTrain)
        listFeatureTrain = listFeatureTrain.loc[:,listFeatureName]
        listFeatureTrain = np.asarray(listFeatureTrain)	
        listLabelTrain = np.asarray(listLabelTrain)	
        clf = getParameterSVM(listFeatureTrain, listLabelTrain)
    return clf

def getParameterSVM(listFeature, listLabel):
    """Return pameters of SVM classifier
    """
    enc = OneHotEncoder()
    enc.fit(listFeature)
    listFeatureSVM = enc.transform(listFeature)	
    clf = svm.LinearSVC()
    clf.fit(listFeatureSVM, listLabel)
    return clf, enc

def ilpSolving(densityMatrix, predicateType):
    shape = np.shape(densityMatrix)
    if(shape[1]!=27):
        densityMatrix = np.insert(densityMatrix,11,-10,axis=1)
    shape = np.shape(densityMatrix)
    prob = LpProblem("SRL", LpMaximize)
    numItem = shape[0]*shape[1]
    densityList = np.reshape(densityMatrix, numItem)
    index = range(0,numItem)
    """
    for i in range(numItem):
        index[i] = str(index[i])
    """
    #create cost dict
    costs = dict(zip(index, densityList))
    #create constrain 1 dict: each argument can take only one type
    tempDict1 = []
    count = 0
    for i in range(shape[0]):
        tempList = [0]*numItem
        tempList[(count):(count+shape[1])] = [1]*shape[1]
        tempDict = dict(zip(index, tempList))
        tempDict1.append(tempDict)
        count += shape[1]
    #create constrain 2 dict: each type appears only one in sentence

    tempDict2 = []
    for i in range(shape[1]):
        if(i==0 or i==1 or i==2 or i==3 or i==4):
            tempList = [0]*numItem
            for j in range(shape[0]):
                tempList[i+j*shape[1]] = 1
            tempDict = dict(zip(index, tempList))
            tempDict2.append(tempDict)

    if(predicateType!=u'V' and predicateType!=u'VP' and predicateType!=u'Vb'):
        tempDict3 = []
        for i in range(shape[1]):
            if(i==2 or i==3 or i==4):
                tempList = [0]*numItem
                for j in range(shape[0]):
                    tempList[i+j*shape[1]] = 1
                tempDict = dict(zip(index, tempList))
                tempDict3.append(tempDict)
    #create variables
    vars = LpVariable.dicts("Var",index,0,1,LpInteger)
    #objective function
    prob += lpSum([costs[i]*vars[i] for i in index]), "Objective Function"
    #constrain 1
    for j in range(shape[0]):
        prob += lpSum([tempDict1[j][i] * vars[i] for i in index]) == 1.0

    #constrain 2
    for j in range(len(tempDict2)):
        prob += lpSum([tempDict2[j][i] * vars[i] for i in index]) <= 1.0

    if(predicateType!=u'V' and predicateType!=u'VP' and predicateType!=u'Vb'):
        for j in range(len(tempDict3)):
            prob += lpSum([tempDict3[j][i] * vars[i] for i in index]) == 0.0
    #solving
    prob.solve()
    listLabel = []
    listVariable = []
    for v in prob.variables():
        listVariable.append(v.varValue)
        if(v.varValue == 1):
            count = int(v.name[4:])
            listLabel.append(count)
    listLabel = sorted(listLabel)
    listLabel = np.asarray(listLabel)
    listLabel = listLabel%27
    listLabel.tolist()
    return listLabel, listVariable

def ilpSolving1(densityMatrix):
    shape = np.shape(densityMatrix)
    if(shape[1]!=27):
        densityMatrix = np.insert(densityMatrix,11,-10,axis=1)
    shape = np.shape(densityMatrix)
    prob = LpProblem("SRL", LpMaximize)
    numItem = shape[0]*shape[1]
    densityList = np.reshape(densityMatrix, numItem)
    densityList = densityList.tolist()
    index = range(0,numItem)
    """
    for i in range(numItem):
        index[i] = str(index[i])
    """
    #create cost dict
    costs = dict(zip(index, densityList))
    #create constrain 1 dict: each argument can take only one type
    tempDict1 = []
    count = 0
    for i in range(shape[0]):
        tempList = [0]*numItem
        tempList[(count):(count+shape[1])] = [1]*shape[1]
        tempDict = dict(zip(index, tempList))
        tempDict1.append(tempDict)
        count += shape[1]
    #create variables
    vars = LpVariable.dicts("Var",index,0,1,LpInteger)
    #objective function
    prob += lpSum([costs[i]*vars[i] for i in index]), "Objective Function"
    #constrain 1
    for j in range(shape[0]):
        prob += lpSum([tempDict1[j][i] * vars[i] for i in index]) == 1.0
    #solving
    prob.solve()
    listLabel = []
    listVariable = []
    for v in prob.variables():
        listVariable.append(v.varValue)
        if(v.varValue == 1):
            count = int(v.name[4:])
            listLabel.append(count)
    listLabel = sorted(listLabel)
    listLabel = np.asarray(listLabel)
    listLabel = listLabel%27
    listLabel.tolist()
    print("Status:", LpStatus[prob.status])
    for v in prob.variables():
        print(v.name, "=", v.varValue)
    print("Total Cost of Ingredients per can = ", value(prob.objective))
    return listLabel, listVariable, densityList, costs, index, vars, tempDict1

def evaluateArg(listLabelTest, listLabelPredict):
    listPrecision = []
    listRecall = []
    listF1Score = []
    listLabelTest = np.asarray(listLabelTest)
    listLabelPredict = np.asarray(listLabelPredict)
    for i in range(27):
        count1 = 0
        index = np.where(listLabelPredict == i)[0]
        for item in index:
            if (listLabelTest[item] == i):
                count1 += 1
        #print count1
        if((listLabelPredict==i).sum()==0):
            precision = 0
        else:
            count2 = (listLabelPredict==i).sum()
            precision = count1/float(count2)
        if((listLabelTest==i).sum()==0):
            recall = 0
        else:
            count3 = (listLabelTest==i).sum()
            recall = count1/float(count3)
        if(precision == 0 and recall == 0):
            f1Score = 0.0
        else:
            f1Score = 2*(precision*recall)/(precision+recall)
        listPrecision.append(precision)
        listRecall.append(recall)
        listF1Score.append(f1Score)
    listPrecision = np.asarray(listPrecision)
    listRecall = np.asarray(listRecall)
    listF1Score = np.asarray(listF1Score)
    #print "ABC"
    return listPrecision, listRecall, listF1Score

####################################################################

def filterData(listID, listTree, listRel, listArg):
    """filter sentences which have empty rel
    """
    newlistID = []
    newlistTree = []
    newlistRel = []
    newlistArg = []
    for i in range(len(listID)):
        if listRel[i][0] == '':
            continue
        newlistID.append(listID[i])
        newlistTree.append(listTree[i])
        newlistArg.append(listArg[i])
        newlistRel.append(listRel[i])
    return newlistID, newlistTree, newlistRel, newlistArg

def makePredicatenode(listID, listTree, listRel, listArg):
    for i in range(len(listTree)):
        id = listID[i]
        tree = listTree[i]
        rels = listRel[i]
        args = listArg[i]
        index = 0
        for rel in rels:
            if rel == '':
                continue
            if len(tree.search_nodes(word = rel)) == 0:
                currentNode = getLeavesPredicate(tree, rel, id)

def _getPath(node, predicateNode):
    ancestor = predicateNode.get_common_ancestor(node)
    path = []
    currentNode = node
    while currentNode != ancestor:
        path.append(reformTag1(currentNode.name))
        path.append('1')
        currentNode = currentNode.up
    path.append(reformTag1(ancestor.name))
    temp = []
    currentNode = predicateNode
    while currentNode != ancestor:
        temp.append(reformTag1(currentNode.name))
        temp.append('0')
        currentNode = currentNode.up
    path.extend(temp[::-1])
    return ''.join(path), len(path)/2+1

def _getHalfPath(path):
    temp = ''
    for c in path:
        if c != '0':
            temp += c
        else:
            break
    return temp

def _getPhraseType(node):
    return reformTag1(node.name)

def _getFunctionType(node):
    return getTagFunction(node.name)

def position(tree, predicateNode):
    dic = {}
    pos = 0
    for leaf in tree:
        if leaf != predicateNode:
            dic[leaf] = pos
        else:
            dic[leaf] = -1
            pos = 1-pos
    return dic

def _getPosition(node, dic):
    for leaf in node.get_leaves():
        if dic[leaf] == 1:
            return 1
    return 0

def _getVoice(tree):
    if len(tree.search_nodes(word = u'bị')) > 0:
        for node in tree.search_nodes(word = u'bị'):
            if node.name == 'V-H':
                for sister in node.get_sisters():
                    if sister.name == 'SBAR':
                        return 0
    if len(tree.search_nodes(word = u'được')) > 0:
        for node in tree.search_nodes(word = u'được'):
            if node.name == 'V-H':
                for sister in node.get_sisters():
                    if sister.name == 'SBAR':
                        return 0
    return 1

def _getHeadWord(node):
    return node.get_leaves()[0].word.strip()

def _getHeadWordType(node):
    return node.get_leaves()[0].name

def _getSubCategorization(predicateNode):
    ancestor = predicateNode.up
    subtree = ancestor.copy()
    for node in subtree.traverse("postorder"):
        node.name = reformTag1(node.name)
    return subtree.write(format = 8)

def includeRel(node, predicateNode):
    return predicateNode in node.get_children()

def onlyOneChildNode(node):
    return len(node.children) == 1

def getWord(node):
    ss = ''
    for leaf in node.get_leaves():
        ss += leaf.word + ' '
    return ss.strip()

def deleteUnderscore(s):
    if s.find('__') != -1:
        return s[2:]
    return s

def getFeatureAllNode(listID, listTree, listRel, listArg, listWordName, listCluster):
    """
    """
    listFeature = []
    listLabel = []
    listCount = []
    for i in range(len(listTree)):
        id = listID[i]
        # print id
        tree = listTree[i]
        for leaf in tree:
            leaf.word = deleteUnderscore(leaf.word)
        listRel[i][0] = deleteUnderscore(listRel[i][0])
        rel = listRel[i][0]
        if rel == '':
            continue
        args = listArg[i]
        # print id
        # print rel
        predicateNode = tree.search_nodes(word = rel)[0]
        # print predicateNode.word.encode('utf8')
        predicateNode.word = deleteUnderscore(predicateNode.word)
        # tree.search_nodes(word = rel)[0].word = deleteUnderscore(tree.search_nodes(word = rel)[0].word)
        rel = deleteUnderscore(rel)
        # listRel[i][0] = deleteUnderscore(listRel[i][0])
        voice = _getVoice(tree)
        positionDic = position(tree, predicateNode)
        """
        found = False
        for j in range(len(listWordName)):
            if rel.lower() == listWordName[j]:
                relNew = listCluster[j]
                found = True
                break
        if not found:
            relNew = '128'
        """
        relNew = rel.lower()
        subcate = _getSubCategorization(predicateNode)
        cnt = 0
        for node in tree.traverse("postorder"):
            if includeRel(node, predicateNode) or onlyOneChildNode(node) or node.name == '':
                continue
            feature = []
            done = False
            feature.append(relNew)
            path, distance = _getPath(node, predicateNode)
            feature.append(path)
            feature.append(_getPhraseType(node))
            feature.append(_getPosition(node, positionDic))
            feature.append(voice)
            feature.append(_getHeadWord(node).lower())
            feature.append(subcate)
            feature.append(_getHalfPath(path))
            feature.append(distance)
            feature.append(_getHeadWordType(node))
            feature.append(_getFunctionType(node))
            feature.append(phraseType(predicateNode.name))
            listFeature.append(feature)
            cnt += 1
            for arg in args:
                if isSame(getWord(node), arg[1]):
                    if reformLabel(arg[0]) == 'ArgM':
                        # print id
                        pass
                    listLabel.append(reformLabel(arg[0]))
                    done = True
                    break
            if not done:
                listLabel.append('None')
        listCount.append(cnt)
    return listLabel, listFeature, listCount

def checkUnderscore(listID, listTree, listRel, listArg):
    print 'checkUnderscore: '
    for i in range(len(listID)):
        id = listID[i]
        # print id
        tree = listTree[i]
        rel = listRel[i][0]
        args = listArg[i]
        if rel.find('__') != -1:
            print rel
            print id
        # print len(tree.search_nodes(word = rel))
        # predicateNode = tree.search_nodes(word = rel)[0])
        # print predicateNode.word.encode('utf8')
        # predicateNode.word = deleteUnderscore(predicateNode.word)
    print '-------------------------->'