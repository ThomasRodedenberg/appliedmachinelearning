from collections import Counter, OrderedDict
from graphviz import Digraph
import matplotlib.pyplot as plt
import math

#fixa info gain v_target antal fuffens och attributes3 uppdelning

class ID3DecisionTreeClassifier :


    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2) :

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')
        self.root = None
        self.nodes = []
        self.att_keys = None

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit


    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'value': None, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                         'classCounts': None, 'nodes': [], 'split_attribute': None}
        self.nodes.append(node)
        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        print(nodeString)
        return


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot


    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self, entropy, data,target,attributes,classes):
        info_gain = []
        i = 0
        for a in attributes:
            info_gain.append(self.info_gain(i,entropy, data,target, attributes[a], classes))
            i+=1
        max_info_gain = max(info_gain)

 
        A = list(attributes.items())[info_gain.index(max_info_gain)][0]
        values = list(attributes.items())[info_gain.index(max_info_gain)][1]
        return A, values

    def get_entropy(self,target,classes):
        entropy = 0

        count = Counter(target)
        n = len(target)
        
        for c in classes:
            p = count[c]/n
            if p != 0:
                entropy += p * math.log(p,2)
        return - entropy

    def info_gain(self,index, entropy, data,target, attribute_vector,classes):
            entropies = []
            info_gain = 0
            for v in attribute_vector:
                v_target = []
                for i in range(len(data)):
                    if v == data[i][index]:
                        v_target.append(target[i])
                if len(v_target) > 0:
                    v_entropy = self.get_entropy(v_target,classes)
                    entropies.append(v_entropy)
                    info_gain = info_gain + (v_entropy * len(v_target) / len(data))
            return entropy - info_gain

    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):

                #root.update({'value': 'hello', 'label': 'okok', 'attribute': None,
        #            'entropy': None, 'samples': None,'classCounts': None, 'nodes': None})
        root = self.new_ID3_node()
        self.root = root
        self.add_node_to_graph(root)
        self.att_keys = list(attributes.keys())
        print('samples ' + str(len(data)))
        tree = self.tree_rek(root,data,target,attributes,classes)
        return tree

    def tree_rek(self,node,data,target,attributes,classes):
        if  len(attributes) == 0 or len(set(target)) == 1 or len(data) == 0:
            node.update({'label': self.common_target(target,classes),'entropy': self.get_entropy(target, classes),'samples': len(data) })
            print('label: ' + str(node['label']))
            return node
        else:
            entropy = self.get_entropy(target,classes)
            A, values = self.find_split_attr(entropy, data,target,attributes,classes)
            
            node.update({'attribute': A, 'entropy':entropy,'samples':len(target), 'classCounts': Counter(target)})
            #print(node['classCounts'])
            
            #For each possible value, v, of A, add a new tree branch below Root, 
            for v in values:
                node_v = self.new_ID3_node()
                self.add_node_to_graph(node_v, node['id'])
                node['nodes'].append(node_v)
                node_v['split_attribute'] = [v, self.att_keys.index(A)]
                data_v = []
                target_v = list()
                for i in range(len(data)):
                    if v == data[i][self.att_keys.index(A)]:
                        data_v.append(data[i])
                        target_v.append(target[i])

                if len(data_v) == 0:
                    node_v.update({'label': self.common_target(target, classes),'samples':0})
                    print('leaf label: ' + str(node_v['label']))
                else:
                    attributes_v = OrderedDict(attributes)
                    attributes_v.pop(A)
                    self.tree_rek(node_v,data_v,target_v,attributes_v,classes)           
        return node
    
    
    def common_target(self, target, classes):
        # Get unique tuples from list
        if len(target) == 0:
            return classes[0]
        b = Counter(target)
        c = b.most_common(1)[0][0]
        #print(c)
        return c


    def predict(self, data, tree) :
        predicted = list()
        #predicted = list()
        for x in data:
            #print(x)
            predicted.append(self.predict_rek(self.root, x))
        return predicted

    def predict_rek(self,node,x):
        if node['label'] is not None:
            return node['label']
        elif len(node['nodes']) > 0:
            for node_v in node['nodes']:
                if node_v['split_attribute'][0] == x[node_v['split_attribute'][1]]:
                    k = self.predict_rek(node_v, x)
                    #if k is not None:
                    return k
                    #else:
                    #   return node['classCounts'].most_common(1)[0][0]
            #print(node['classCounts'])
            return node['classCounts'].most_common(1)[0][0]
        #kolla vanligaste labels för barnen till noden med None, extra attribut i nodes uppdateras i
                #tree_rek och kollas i predict_rek


            
