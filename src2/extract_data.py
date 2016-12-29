import lib
from ete2 import Tree
import codecs
from collections import Counter
import os
import re


dataFile = 'data.xml'
# dataFile = 'dev.xml'


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
                    if dataFile == "dev.xml":
                        print "temp: ", temp, ' | ', leaves[j].word
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


def get_representative_node(nodes):
    representative_node = nodes[0]
    for node in nodes:
        representative_node = representative_node.get_common_ancestor(node)
    return representative_node


def use_representive_node():
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
        if dataFile == "dev.xml":
            tree_str = tree.get_ascii().split('\n')
            words = list()
            for leaf in tree:
                words.append(leaf.word)

            for i in range(len(tree_str)):
                if i % 2 == 1:
                    tree_str[i] += ' ' + words[i/2]
            for line in tree_str:
                print line
            print 'Rel: ', rel[0]
            for arg in args:
                print "Arg: ", arg[0], ' -> ', arg[1]

        tag = ''
        BOI_tags = dict()
        BOI_phrase_type = dict()
        BOI_functional_tag = dict()
        rel_nodes = inTree(tree, rel[0])
        representative_rel_node = get_representative_node(rel_nodes)
        # print rel[0]
        # print representative_rel_node
        for node in rel_nodes:
            BOI_tags[node] = ['Rel', 0]
            BOI_phrase_type[node] = [lib.reformTag1(representative_rel_node.name), 0]
            BOI_functional_tag[node] = [lib.getTagFunction(representative_rel_node.name), 0]
        for i, arg in enumerate(args):
            if len(arg[1]) == 0:
                continue
            # print arg[1]
            arg_nodes = inTree(tree, arg[1])
            if len(arg_nodes) > 0:
                representative_arg_node = get_representative_node(arg_nodes)
                args_collection.append(lib.reformLabel(arg[0]))
                for node in arg_nodes:
                    BOI_tags[node] = [lib.reformLabel(arg[0]), i+1]
                    BOI_phrase_type[node] = [lib.reformTag1(representative_arg_node.name), 0]
                    BOI_functional_tag[node] = [lib.getTagFunction(representative_arg_node.name), 0]
        # for node in BOI_tags:
        #     print node, ' -> ', BOI_tags[node]
        leaves = tree.get_leaves()
        for i, leaf in enumerate(leaves):
            if BOI_tags.get(leaf) is None:
                leaf.add_features(prop='O')
            else:
                if i > 0 and BOI_tags.get(leaves[i-1]) is not None \
                        and BOI_tags.get(leaves[i])[0] == BOI_tags.get(leaves[i-1])[0] \
                        and BOI_tags.get(leaves[i])[1] != BOI_tags.get(leaves[i-1])[1]:
                    leaf.add_features(prop='B-' + BOI_tags[leaf][0])
                else:
                    leaf.add_features(prop='I-' + BOI_tags[leaf][0])
            if BOI_phrase_type.get(leaf) is None:
                leaf.add_features(phrase="O")
            else:
                if i > 0 and BOI_phrase_type.get(leaves[i-1]) is not None \
                        and BOI_phrase_type.get(leaves[i])[0] == BOI_phrase_type.get(leaves[i-1])[0] \
                        and BOI_phrase_type.get(leaves[i])[1] != BOI_phrase_type.get(leaves[i-1])[1]:
                    leaf.add_features(phrase='B-' + BOI_phrase_type[leaf][0])
                else:
                    leaf.add_features(phrase='I-' + BOI_phrase_type[leaf][0])
            if BOI_functional_tag.get(leaf) is None:
                leaf.add_features(functag="O")
            else:
                if i > 0 and BOI_functional_tag.get(leaves[i-1]) is not None \
                        and BOI_functional_tag.get(leaves[i])[0] == BOI_functional_tag.get(leaves[i-1])[0] \
                        and BOI_functional_tag.get(leaves[i])[1] != BOI_functional_tag.get(leaves[i-1])[1]:
                    leaf.add_features(functag='B-' + BOI_functional_tag[leaf][0])
                else:
                    leaf.add_features(functag='I-' + BOI_functional_tag[leaf][0])
        nodes = []
        for node in tree.traverse("preorder"):
            nodes.append(node)
        cnt = 0
        voice = lib.getVoice(tree, rel[0])
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
                        line = node.word + '\t' + node.name + '\t' + str(voice) + '\t' + node.phrase + '\t' + node.functag + '\t' + tag + '\t' + node.prop
                        if writable:
                            writer.write(line + '\n')
                        tag = ''
                    else:
                        print node.name
        if writable:
            writer.write('\n')
    if writable:
        writer.close()
    # counter = Counter(args_collection)
    # for arg in sorted(counter.keys()):
    #     print arg, ' -> ', counter[arg]


def main():
    kfold = 5

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

    writers = list()
    for _ in range(kfold):
        writers.append(codecs.open('set/set_' + str(_) + '.txt', 'w', 'utf8'))
    args_collection = list()
    stt = 0
    for tree, rel, args, _id in zip(listTreeTotal, listRelTotal, listArgTotal, listIDTotal):
        tag = ''
        BOI_tags = dict()
        BOI_phrase_type = dict()
        BOI_functional_tag = dict()
        rel_nodes = inTree(tree, rel[0])
        # print rel[0]
        # print representative_rel_node
        for node in rel_nodes:
            BOI_tags[node] = ['Rel', 0]
        for i, arg in enumerate(args):
            if len(arg[1]) == 0:
                continue
            # print arg[1]
            arg_nodes = inTree(tree, arg[1])
            if len(arg_nodes) > 0:
                args_collection.append(lib.reformLabel(arg[0]))
                for node in arg_nodes:
                    BOI_tags[node] = [lib.reformLabel(arg[0]), i + 1]
        # for node in BOI_tags:
        #     print node, ' -> ', BOI_tags[node]
        leaves = tree.get_leaves()
        phrase_nodes = list()
        for leaf in leaves:
            phrase_node = leaf
            while phrase_node != tree and not lib.isPhraseType(phrase_node.name):
                phrase_node = phrase_node.up
            if lib.isPhraseType(phrase_node.name) and phrase_node not in phrase_nodes:
                phrase_nodes.append(phrase_node)
        for i, node in enumerate(phrase_nodes):
            phrase_tag = lib.reformTag1(node.name)
            func_tag = lib.getTagFunction(node.name)
            for leaf in node:
                BOI_phrase_type[leaf] = [phrase_tag, i]
                if func_tag != 'None':
                    BOI_functional_tag[leaf] = [func_tag, i]
        for i, leaf in enumerate(leaves):
            if BOI_tags.get(leaf) is None:
                leaf.add_features(prop='O')
            else:
                if i > 0 and BOI_tags.get(leaves[i - 1]) is not None \
                        and BOI_tags.get(leaves[i])[0] == BOI_tags.get(leaves[i - 1])[0] \
                        and BOI_tags.get(leaves[i])[1] != BOI_tags.get(leaves[i - 1])[1]:
                    leaf.add_features(prop='B-' + BOI_tags[leaf][0])
                else:
                    leaf.add_features(prop='I-' + BOI_tags[leaf][0])
            if BOI_phrase_type.get(leaf) is None:
                leaf.add_features(phrase="O")
            else:
                if i > 0 and BOI_phrase_type.get(leaves[i - 1]) is not None \
                        and BOI_phrase_type.get(leaves[i])[0] == BOI_phrase_type.get(leaves[i - 1])[0] \
                        and BOI_phrase_type.get(leaves[i])[1] != BOI_phrase_type.get(leaves[i - 1])[1]:
                    leaf.add_features(phrase='B-' + BOI_phrase_type[leaf][0])
                else:
                    leaf.add_features(phrase='I-' + BOI_phrase_type[leaf][0])
            if BOI_functional_tag.get(leaf) is None:
                leaf.add_features(functag="O")
            else:
                if i > 0 and BOI_functional_tag.get(leaves[i - 1]) is not None \
                        and BOI_functional_tag.get(leaves[i])[0] == BOI_functional_tag.get(leaves[i - 1])[0] \
                        and BOI_functional_tag.get(leaves[i])[1] != BOI_functional_tag.get(leaves[i - 1])[1]:
                    leaf.add_features(functag='B-' + BOI_functional_tag[leaf][0])
                else:
                    leaf.add_features(functag='I-' + BOI_functional_tag[leaf][0])
        nodes = []
        for node in tree.traverse("preorder"):
            nodes.append(node)
        cnt = 0
        voice = lib.getVoice(tree, rel[0])
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
                            sisters = nodes[i + 1].get_sisters()
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
                        line = node.word + '\t' + node.name + '\t' + str(voice) + '\t' + rel[0] + '\t' + node.phrase + '\t' + node.functag + '\t' + tag + '\t' + node.prop
                        writers[stt%kfold].write(line + '\n')
                        tag = ''
                    else:
                        print node.name
        writers[stt%kfold].write('\n')
        stt += 1
    for _ in range(kfold):
        writers[_].close()


def make_data_folds():
    kfold = 5
    data_folds = [list() for _ in range(kfold)]
    for i in range(kfold):
        data_file = 'set/set_' + str(i) + '.txt'
        for line in codecs.open(data_file, 'r', 'utf8'):
            line = line.strip()
            data_folds[i].append(line)
    for i in range(kfold):
        if not os.path.exists('set/' + str(i)):
            os.mkdir('set/' + str(i))
        train_writer = codecs.open('set/' + str(i) + '/train.txt', 'w', 'utf8')
        test_writer = codecs.open('set/' + str(i) + '/test.txt', 'w', 'utf8')
        for j in range(kfold):
            if i == j:
                for line in data_folds[j]:
                    test_writer.write(line)
                    test_writer.write('\n')
            else:
                for line in data_folds[j]:
                    train_writer.write(line)
                    train_writer.write('\n')
        train_writer.close()
        test_writer.close()


if __name__ == '__main__':
    # main()
    make_data_folds()
