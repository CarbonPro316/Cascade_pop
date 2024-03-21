import pickle
import time
import networkx as nx
from absl import app, flags
from utils.graphwave.graphwave import *

# flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('emb_dim', 64, 'Embedding dimension.')
flags.DEFINE_integer('max_seq', 100, 'Max length of cascade sequence.')
flags.DEFINE_integer('num_s', 2, 'Number of s for spectral graph wavelets.')
flags.DEFINE_integer('t_o', 3600, 'Observation time.')
flags.DEFINE_string('s_weight', 'none', 'soical information.')

# paths
flags.DEFINE_string( 'input', '../datasets/weibo/', 'Pre-training data path.')


def main(argv):
    # hyper-params
    emb_dim = FLAGS.emb_dim
    max_seq = FLAGS.max_seq
    NUM_S = FLAGS.num_s
    observation_time = FLAGS.t_o
    s_weight=FLAGS.s_weight
    # hyper-params

    def sequence2list(filename):
        graphs = dict()
        with open(filename, 'r') as f:
            for line in f:
                paths = line.strip().split('\t')[:-1][:max_seq + 1]
                graphs[paths[0]] = list()
                for i in range(1, len(paths)):
                    nodes = paths[i].split(":")[0]
                    time = paths[i].split(":")[1]
                    graphs[paths[0]] \
                        .append([[int(x) for x in nodes.split(",")],
                                 int(time)])

        return graphs

    def read_labels(filename):
        labels = dict()
        with open(filename, 'r') as f:
            for line in f:
                id = line.strip().split('\t')[0]
                labels[id] = line.strip().split('\t')[-1]
        return labels

    def write_cascade(graphs, labels, filename, s_weight,print_note='Write cascade graphs', weight=False):
        """
        Input: cascade graphs, global embeddings
        Output: cascade embeddings, with global embeddings appended
        """
        y_data = list()
        new_input = list()
        cascade_i = 0
        cascade_size = len(graphs)
        total_time = 0
        
        # for each cascade graph, generate its embeddings via wavelets
        for key, graph in graphs.items():
            start_time = time.time()
            y = int(labels[key])

            # list for saving embeddings
            new_temp = list()

            # build graph
            g = nx.Graph()
            nodes_index = list()
            list_edge = list()
            cascade_embedding = list()
            times = list()
            #t_o = observation_time

            # add edges into graph
            for path in graph:
                t = path[1]
                #if t >= t_o:
                #    continue
                nodes = path[0]
                if len(nodes) == 1:
                    nodes_index.extend(nodes)
                    times.append(1)
                    continue
                else:
                    nodes_index.extend([nodes[-1]])
                #if weight:
                #    edge = (nodes[-1], nodes[-2], (1 - t / t_o))
                #    times.append(1 - t / t_o)
                #else:
                edge = (nodes[-1], nodes[-2])
                list_edge.append(edge)
            if weight:
                g.add_weighted_edges_from(list_edge)
            else:
                g.add_edges_from(list_edge)

            # this list is used to make sure the node order of `chi` is same to node order of `cascade`
            nodes_index_unique = list(set(nodes_index))
            nodes_index_unique.sort(key=nodes_index.index)
            

            if g.number_of_nodes() <= 1:
                continue
            # embedding dim check
            d = emb_dim / (2 * NUM_S)
            if emb_dim % (2 * NUM_S) != 0:
                raise ValueError

            # generate embeddings
            chi, _, _ = graphwave_alg(g, np.linspace(0, 100, int(d)),
                                      taus='auto', verbose=False,
                                      nodes_index=nodes_index_unique,
                                      nb_filters=NUM_S)
            # save embeddings into list
            if s_weight=='deepcas':
                for node in nodes_index:
                    emb=np.append(chi[nodes_index_unique.index(node)],s_emb[str(node)])
                    cascade_embedding.append(emb)
            elif s_weight=='android':
                for node in nodes_index:
                    emb=np.append(chi[nodes_index_unique.index(node)],s_emb.get(str(node),1/3741))
                    cascade_embedding.append(emb)
            elif s_weight=='christianity':
                for node in nodes_index:
                    emb=np.append(chi[nodes_index_unique.index(node)],s_emb.get(str(node),1/1003))
                    cascade_embedding.append(emb)
            else:
                for node in nodes_index:
                    cascade_embedding.append(chi[nodes_index_unique.index(node)])
            # concat node features to node embedding
            if weight:
                cascade_embedding = np.concatenate([np.reshape(times, (-1, 1)), np.array(cascade_embedding)[:, 1:]],
                                                   axis=1)

            # save embeddings
            new_temp.extend(cascade_embedding)
            new_input.append(new_temp)

            # save label
            y_data.append(y)

            # log
            total_time += time.time() - start_time
            cascade_i += 1
            if cascade_i % 100 == 0:
                speed = total_time / cascade_i
                eta = (cascade_size - cascade_i) * speed
                print("{}, {}/{}, eta: {:.2f} minutes".format(
                    print_note, cascade_i, cascade_size, eta / 60))
        # save embeddings and labels into file
        with open(filename, 'wb') as fin:
            pickle.dump((new_input, y_data), fin)

    def write_aug_cascade(graphs, filename,s_weight, print_note='augmentation 1', weight=False):
        """
        Input: cascade graphs, global embeddings
        Output: cascade embeddings, with global embeddings appended
        """
        new_input = list()
        embedding_size = emb_dim
        cascade_i = 0
        cascade_size = len(graphs)
        total_time = 0

        # for each cascade graph, generate its embeddings via wavelets
        for key, graph in graphs.items():
            start_time = time.time()
            new_temp = list()

            # build graph
            g = nx.Graph()
            nodes_index = list()
            list_edge = list()
            cascade_embedding = list()
            times = list()
            #t_o = observation_time

            # add edges into graph
            for path in graph:
                t = path[1]
                #if t >= t_o:
                #    continue
                nodes = path[0]
                if len(nodes) == 1:
                    nodes_index.extend(nodes)
                    times.append(1)
                    g.add_node(nodes[-1])
                    continue
                else:
                    nodes_index.extend([nodes[-1]])
                #if weight:
                #    edge = (nodes[-1], nodes[-2], (1 - t / t_o))
                #    times.append(1 - t / t_o)
                #else:
                edge = (nodes[-1], nodes[-2])
                list_edge.append(edge)
            if weight:
                g.add_weighted_edges_from(list_edge)
            else:
                g.add_edges_from(list_edge)

            # this list is used to make sure the node order of `chi` is same to node order of `cascade`
            nodes_index_unique = list(set(nodes_index))
            nodes_index_unique.sort(key=nodes_index.index)

            if g.number_of_nodes() <= 1:
                continue

            # embedding dim check
            d = embedding_size / (2 * NUM_S)
            if embedding_size % (2 * NUM_S) != 0:
                raise ValueError

            # generate embeddings
            chi, _, _ = graphwave_alg(g, np.linspace(0, 100, int(d)),
                                      taus='auto', verbose=False,
                                      nodes_index=nodes_index_unique,
                                      nb_filters=NUM_S)
            # save embeddings into list
            if s_weight=='deepcas':
                for node in nodes_index:
                    emb=np.append(chi[nodes_index_unique.index(node)],s_emb[str(node)])
                    cascade_embedding.append(emb)
            elif s_weight=='android':
                for node in nodes_index:
                    emb=np.append(chi[nodes_index_unique.index(node)],s_emb.get(str(node),1/3741))
                    cascade_embedding.append(emb)
            elif s_weight=='christianity':
                for node in nodes_index:
                    emb=np.append(chi[nodes_index_unique.index(node)],s_emb.get(str(node),1/1003))
                    cascade_embedding.append(emb)
            else:
                for node in nodes_index:
                    cascade_embedding.append(chi[nodes_index_unique.index(node)])
            # concat node features to node embedding
            if weight:
                cascade_embedding = np.concatenate([np.reshape(times, (-1, 1)), np.array(cascade_embedding)[:, 1:]],
                                                   axis=1)

            # save embeddings
            new_temp.extend(cascade_embedding)
            new_input.append(new_temp)

            # log
            total_time += time.time() - start_time
            cascade_i += 1
            if cascade_i % 100 == 0:
                speed = total_time / cascade_i
                eta = (cascade_size - cascade_i) * speed
                print("{}, {}/{}, eta: {:.2f} minutes".format(
                    print_note, cascade_i, cascade_size, eta / 60))

        # save embeddings into file
        with open(filename, 'wb') as fin:
            pickle.dump(new_input, fin)

    time_start = time.time()

    #get social information
    
    s_emb={}
    if s_weight=='deepcas':
        filename="D:\Code\ccgl_pytorch\datasets\deepcas\global_graph.txt"
        with open(filename, 'r') as f:
            for line in f:
                parts=line.split("\t")
                if parts[2]=="null\n":
                    s_emb[parts[0]]=0
                else:
                    weight=0
                    for i in range(2,len(parts)-1):
                        node_weight=parts[i].split(':')[1]
                        weight=weight+int(node_weight)
                    weight=weight+int(parts[-1][-2])
                    s_emb[parts[0]]=weight
        max=0
        for k,v in s_emb.items():
            if v>max:
                max=v
        for k,v in s_emb.items():
            s_emb[k]=v/max
    elif s_weight=='android':
        filename="D:/Code/ccgl_pytorch/datasets/android/graph.txt"
        S_G=nx.Graph()
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                parts=line.split()
                S_G.add_edge(parts[0],parts[1])
        max=-1
        for k,v in S_G.degree:
            if v>max:
                max=v
        for k,v in S_G.degree:
            s_emb[k]=v/max
    elif s_weight=='christianity':
        filename="D:/Code/ccgl_pytorch/datasets/christianity/graph.txt"
        S_G=nx.Graph()
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                parts=line.split()
                S_G.add_edge(parts[0],parts[1])
        max=-1
        for k,v in S_G.degree:
            if v>max:
                max=v
        for k,v in S_G.degree:
            s_emb[k]=v/max
    else:
        print("no social information")

    # get the information of nodes/users of cascades
    graphs_train = sequence2list(FLAGS.input + 'train.txt')
    graphs_val = sequence2list(FLAGS.input + 'val.txt')
    graphs_test = sequence2list(FLAGS.input + 'test.txt')
    graphs_aug_1_04 = sequence2list(FLAGS.input + 'aug_1_0.4.txt')
    graphs_aug_1_06 = sequence2list(FLAGS.input + 'aug_1_0.6.txt')
    graphs_aug_1_08 = sequence2list(FLAGS.input + 'aug_1_0.8.txt')
    graphs_aug_2_05 = sequence2list(FLAGS.input + 'aug_2_0.5.txt')
    graphs_aug_2_07 = sequence2list(FLAGS.input + 'aug_2_0.7.txt')
    graphs_aug_2_085 = sequence2list(FLAGS.input + 'aug_2_0.85.txt')
    graphs_unlabel_aug_1_04 = sequence2list(FLAGS.input + 'unlabel_aug_1_0.4.txt')
    graphs_unlabel_aug_1_06 = sequence2list(FLAGS.input + 'unlabel_aug_1_0.6.txt')
    graphs_unlabel_aug_1_08 = sequence2list(FLAGS.input + 'unlabel_aug_1_0.8.txt')
    graphs_unlabel_aug_2_05 = sequence2list(FLAGS.input + 'unlabel_aug_2_0.5.txt')
    graphs_unlabel_aug_2_07 = sequence2list(FLAGS.input + 'unlabel_aug_2_0.7.txt')
    graphs_unlabel_aug_2_085 = sequence2list(FLAGS.input + 'unlabel_aug_2_0.85.txt')

    # get the information of labels and sizes of cascades
    label_train = read_labels(FLAGS.input + 'train.txt')
    label_val = read_labels(FLAGS.input + 'val.txt')
    label_test = read_labels(FLAGS.input + 'test.txt')

    print("Start writing train set into file.")
    write_cascade(graphs_train, label_train, FLAGS.input + 'train.pkl',s_weight, print_note='(1/7) Write train set')
    print("Start writing validation set into file.")
    write_cascade(graphs_val, label_val, FLAGS.input + 'val.pkl',s_weight, print_note='(2/7) Write val set')
    print("Start writing test set into file.")
    write_cascade(graphs_test, label_test, FLAGS.input + 'test.pkl', s_weight,print_note='(3/7) Write test set')
    print("Start writing aug set 1 into file.")
    write_aug_cascade(graphs_aug_1_04, FLAGS.input + 'data_aug_1_04.pkl', s_weight,print_note='(4/7) Write augmented cascade graphs')
    write_aug_cascade(graphs_aug_1_06, FLAGS.input + 'data_aug_1_06.pkl', s_weight,print_note='(4/7) Write augmented cascade graphs')
    write_aug_cascade(graphs_aug_1_08, FLAGS.input + 'data_aug_1_08.pkl', s_weight,print_note='(4/7) Write augmented cascade graphs')
    print("Start writing aug set 2 into file.")
    write_aug_cascade(graphs_aug_2_05, FLAGS.input + 'data_aug_2_05.pkl', s_weight,print_note='(5/7) Write augmented cascade graphs')
    write_aug_cascade(graphs_aug_2_07, FLAGS.input + 'data_aug_2_07.pkl', s_weight,print_note='(5/7) Write augmented cascade graphs')
    write_aug_cascade(graphs_aug_2_085, FLAGS.input + 'data_aug_2_085.pkl', s_weight,print_note='(5/7) Write augmented cascade graphs')
    print("Start writing unlabel aug set 1 into file.")
    write_aug_cascade(graphs_unlabel_aug_1_04, FLAGS.input + 'data_unlabel_aug_1_04.pkl',s_weight,
                      print_note='(6/7) Write unlabeled augmented cascade graphs')
    write_aug_cascade(graphs_unlabel_aug_1_06, FLAGS.input + 'data_unlabel_aug_1_06.pkl',s_weight,
                      print_note='(6/7) Write unlabeled augmented cascade graphs')
    write_aug_cascade(graphs_unlabel_aug_1_08, FLAGS.input + 'data_unlabel_aug_1_08.pkl',s_weight,
                      print_note='(6/7) Write unlabeled augmented cascade graphs')
    print("Start writing unlabel aug set 2 into file.")
    write_aug_cascade(graphs_unlabel_aug_2_05, FLAGS.input + 'data_unlabel_aug_2_05.pkl',s_weight,
                      print_note='(7/7) Write unlabeled augmented cascade graphs')
    write_aug_cascade(graphs_unlabel_aug_2_07, FLAGS.input + 'data_unlabel_aug_2_07.pkl',s_weight,
                      print_note='(7/7) Write unlabeled augmented cascade graphs')
    write_aug_cascade(graphs_unlabel_aug_2_085, FLAGS.input + 'data_unlabel_aug_2_085.pkl',s_weight,
                      print_note='(7/7) Write unlabeled augmented cascade graphs')

    time_end = time.time()
    print("Processing time: {0:.2f}s".format(time_end - time_start))


if __name__ == "__main__":
    app.run(main)
