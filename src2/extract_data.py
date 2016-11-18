import lib
from ete2 import Tree


def inTree(ptree, ss):
    if ss == '':
        return []
    if len(ptree.search_nodes(word = ss)) > 0:
        return ptree.search_nodes(word = ss)
    else:
        leaves = ptree.get_leaves()
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


dataFile = 'data.xml'
listSentence, listID, listCDATA = lib.readData(dataFile)
listTag, listWord = lib.convertData(listSentence)
listTagClone, listWordClone = lib.convertData(listSentence)
listTree = lib.dataToTree(listTagClone, listWordClone)

listRel, listArg = lib.readCDATA(listCDATA, listWord, listID)
listID1Rel, listTree1Rel, listRel1Rel, listArg1Rel = lib.collectTree1Rel(listID, listTree, listRel, listArg)
listIDExtractFromMutliRel, listTreeExtractFromMutliRel, listRelExtractFromMutliRel, listArgExtractFromMutliRel = \
    lib.extractFromMultiRel(listID, listTree, listRel, listArg)
listIDTotal, listTreeTotal, listRelTotal, listArgTotal = \
    lib.mergeData(listID1Rel, listTree1Rel, listRel1Rel, listArg1Rel,
                  listIDExtractFromMutliRel, listTreeExtractFromMutliRel,
                  listRelExtractFromMutliRel, listArgExtractFromMutliRel)

for tree, rel in zip(listTreeTotal, listRelTotal):
    leaves = tree.get_leaves()
    for node in tree.traverse("preorder"):
        print node.name
    break
