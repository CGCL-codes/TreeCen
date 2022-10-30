'''
This python file is used to get six centralities of each sourch code.
'''

import json
import numpy as np
import javalang
from javalang.ast import Node
from anytree import AnyNode, RenderTree
import networkx as nx
import glob
import pandas as pd
from  tqdm  import  tqdm 

'''
There are 57 node types and 15 token types, which are numbered for ease of analysis.
'''

nodetypedict = {'MethodDeclaration': 0, 'Modifier': 1, 'FormalParameter': 2, 'ReferenceType': 3, 'BasicType': 4,
     'LocalVariableDeclaration': 5, 'VariableDeclarator': 6, 'MemberReference': 7, 'ArraySelector': 8, 'Literal': 9,
     'BinaryOperation': 10, 'TernaryExpression': 11, 'IfStatement': 12, 'BlockStatement': 13, 'StatementExpression': 14,
     'Assignment': 15, 'MethodInvocation': 16, 'Cast': 17, 'ForStatement': 18, 'ForControl': 19,
     'VariableDeclaration': 20, 'TryStatement': 21, 'ClassCreator': 22, 'CatchClause': 23, 'CatchClauseParameter': 24,
     'ThrowStatement': 25, 'WhileStatement': 26, 'ArrayInitializer': 27, 'ReturnStatement': 28, 'Annotation': 29,
     'SwitchStatement': 30, 'SwitchStatementCase': 31, 'ArrayCreator': 32, 'This': 33, 'ConstructorDeclaration': 34,
     'TypeArgument': 35, 'EnhancedForControl': 36, 'SuperMethodInvocation': 37, 'SynchronizedStatement': 38,
     'DoStatement': 39, 'InnerClassCreator': 40, 'ExplicitConstructorInvocation': 41, 'BreakStatement': 42,
     'ClassReference': 43, 'SuperConstructorInvocation': 44, 'ElementValuePair': 45, 'AssertStatement': 46,
     'ElementArrayValue': 47, 'TypeParameter': 48, 'FieldDeclaration': 49, 'SuperMemberReference': 50,
     'ContinueStatement': 51, 'ClassDeclaration': 52, 'TryResource': 53, 'MethodReference': 54,
     'LambdaExpression': 55, 'InferredFormalParameter': 56}
tokendict = {'DecimalInteger': 57, 'HexInteger': 58, 'Identifier': 59, 'Keyword': 60, 'Modifier': 61, 'Null': 62,
              'OctalInteger': 63, 'Operator': 64, 'Separator': 65, 'String': 66, 'Annotation': 67, 'BasicType': 68,
              'Boolean': 69, 'DecimalFloatingPoint': 70, 'HexFloatingPoint': 71}



'''
get_token function
--------------------
This function is used to get token from each node of ast.

#Arguments
    node: The node of ast
'''
def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token

'''
get_child function
--------------------
This function is used to get the child nodes of root nodes

#Argument
    root: The root nodes of each

'''
def get_child(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item
    return list(expand(children))
'''
createtree function
--------------------
This function is used to get the tree structure of the source code

#Arguments
    root: The root of the tree structure
    node: The output of parser.parse_member_declaration()
    nodelist: The list of nodes that already be processed
    parent: whether the argument node has the parent node

'''
def createtree(root, node, nodelist, parent=None):
    id = len(nodelist)
    token, children = get_token(node), get_child(node)
    if id == 0:
        #if it is the root node
        root.token = token
        root.data = node
    else:
        newnode = AnyNode(id=id, token=token, data=node, parent=parent)
    nodelist.append(node)
    for child in children:
        if id == 0:
            createtree(root, child, nodelist, parent=root)
        else:
            createtree(root, child, nodelist, parent=newnode)



'''
getnodeandedge function
--------------------
Get all the edges in the tree
the src[i] is the starting point of the edge[i] while tgt[i] is the ending nodes of edge[i]

#Argument
    node: The node in the ast that need to be processed
    src: The set of starting nodes of edges"
    tgt: The set of ending nodes of edges"

'''

def getnodeandedge(node, src, tgt):
    for child in node.children:
        src.append(node.token)
        tgt.append(child.token)
        getnodeandedge(child, src, tgt)


'''
get_single_cent_matrix function
--------------------
This function is used to get six centrlities of each source code

#Argument
    path: The absolute path of the source code

'''
def get_single_cent_matrix(path):
    g=nx.Graph()
    # generate ast and token
    with open(path, 'r', encoding='utf-8') as src_file:
        programtext = src_file.read()
        tokens = list(javalang.tokenizer.tokenize(programtext))
    try:
        programtokens = javalang.tokenizer.tokenize(programtext)
        parser = javalang.parse.Parser(programtokens)
        tree = parser.parse_member_declaration()
    except:
        return
    
    nodelist = []
    newtree = AnyNode(id=0, token=None, data=None)
    createtree(newtree, tree, nodelist)

    # generate the typedict
    typedict = {}
    for token in tokens:
        token_type = str(type(token))[:-2].split(".")[-1]
        token_value = token.value
        if token_value not in typedict:
            typedict[token_value] = token_type

    src = []
    tgt = []
    getnodeandedge(newtree, src, tgt)

    #Add nodes to the graph
    for item in nodetypedict.items():
        g.add_node(item[1],name = item[0])
    for item in tockendict.items():
        g.add_node(item[1],name = item[0])
    
    #Add edges to the graph
    for i in range(len(src)):
        m = nodetypedict[src[i]]
        name = tgt[i]
        try:
            n = nodetypedict[name]
        except KeyError:
            try:
                n = tockendict[typedict[name]]
            except KeyError:
                n = 62
        if g.has_edge(m, n):
                g[m][n]['weight'] += 1
        else:
            g.add_edge(m, n, weight=1)
    this_all_cents = dict()
    try:
        this_all_cents["cent_harm"] = [cent /len(g) for cent in nx.harmonic_centrality(g).values()]
        this_all_cents["cent_eigen"] = [cent for cent in nx.eigenvector_centrality(g).values()]
        this_all_cents["cent_close"] = [cent for cent in nx.closeness_centrality(g).values()]
        this_all_cents["cent_between"] = [cent for cent in nx.betweenness_centrality(g).values()]
        this_all_cents["cent_degree"] = [cent for cent in nx.degree_centrality(g).values()]
        this_all_cents["cent_katz"] = [cent for cent in nx.katz_centrality(g).values()]
    except:
        return
    return this_all_cents

'''
#Argument
    java_src_path: the path of source codes
    outpath: the path of output dictionary
'''
def main():
    java_src_path = '/Users/apple/Desktop/code-clone/id2sourcecode/'
    out_path = '/Users/apple/Desktop/code-clone/cent_matrix1.json'
    javalist = glob.glob(java_src_path+'*.java')
    all_cent_mtxs = dict()
    for java_file in tqdm(javalist):
        file_id = java_file.rsplit("/")[-1][:-5]
        cent_metrix_dict = get_single_cent_matrix(java_file)
        if cent_metrix_dict == None:
            continue
        all_cent_mtxs[file_id] = cent_metrix_dict
    with open(out_path, 'w') as f:
        json.dump(all_cent_mtxs, f)


if __name__ == '__main__':
    main()
