import lib
from ete2 import Tree
import codecs
from collections import Counter


def inTree(ptree, ss):
    if ss == '':
        return []
    if len(ptree.search_nodes(word=ss)) > 0:
        return ptree.search_nodes(word=ss)
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
                            return leaves[i:j]
                        else:
                            temp = ''
                if ss == temp:
                    return leaves[i:]
    return []


def is_all_star(ss):
    for c in ss:
        if c != '*':
            return False
    return True


def main():
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

    writable = True
    if writable:
        writer = codecs.open('conll.txt', 'w', 'utf8')
    args_collection = list()
    for tree, rel, args, _id in zip(listTreeTotal, listRelTotal, listArgTotal, listIDTotal):
        # tree_str = tree.get_ascii().split('\n')
        # words = list()
        # for leaf in tree:
        #     words.append(leaf.word)
        #
        # for i in range(len(tree_str)):
        #     if i % 2 == 1:
        #         tree_str[i] += ' ' + words[i/2]
        # for line in tree_str:
        #     print line
        tag = ''
        BOI_tags = dict()
        rel_nodes = inTree(tree, rel[0])
        for node in rel_nodes:
            BOI_tags[node] = ['Rel', 0]
        for i, arg in enumerate(args):
            # print arg[1]
            arg_nodes = inTree(tree, arg[1])
            args_collection.append(lib.reformLabel(arg[0]))
            for node in arg_nodes:
                BOI_tags[node] = [lib.reformLabel(arg[0]), i+1]
        # for node in BOI_tags:
        #     print node, ' -> ', BOI_tags[node]
        leaves = tree.get_leaves()
        for i, leaf in enumerate(leaves):
            if BOI_tags.get(leaf) is None:
                leaf.add_features(prop='O')
            else:
                if i > 0 and BOI_tags.get(leaves[i-1]) is not None and BOI_tags.get(leaves[i])[0] == BOI_tags.get(leaves[i-1])[0] and BOI_tags.get(leaves[i])[1] != BOI_tags.get(leaves[i-1])[1]:
                    leaf.add_features(prop='B-' + BOI_tags[leaf][0])
                else:
                    leaf.add_features(prop='I-' + BOI_tags[leaf][0])
        nodes = []
        for node in tree.traverse("preorder"):
            nodes.append(node)
        cnt = 0
        for i, node in enumerate(nodes):
            if lib.isPhraseType(node.name) or lib.isSType(node.name):
                tag += '(' + node.name
                cnt += 1
            else:
                if not node.is_root():
                    tag += '*'
                    if node.is_leaf():
                        if is_all_star(tag):
                            tag = '*'
                        if i < len(nodes) - 1:
                            curNode = node
                            sisters = nodes[i+1].get_sisters()
                            while True:
                                if curNode in sisters:
                                    break
                                tag += ')'
                                cnt -= 1
                                curNode = curNode.up
                        else:
                            for _ in range(cnt):
                                tag += ')'
                            cnt = 0
                        line = node.word + '\t' + node.name + '\t' + tag + '\t' + node.prop
                        if writable:
                            writer.write(line + '\n')
                        tag = ''
                    else:
                        print node.name
        if writable:
            writer.write('\n')
    if writable:
        writer.close()
    counter = Counter(args_collection)
    for arg in sorted(counter.keys()):
        print arg, ' -> ', counter[arg]

if __name__ == '__main__':
    main()
