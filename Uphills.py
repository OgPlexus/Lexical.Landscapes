
# coding: utf-8

# ### Search for Uphills

# In[56]:


get_ipython().run_cell_magic('time', '', 'import numpy as np\nfrom scipy.integrate import odeint\nimport matplotlib as mpl\nimport pandas as pd\nimport pylab as p\nimport os\nimport csv\n\nn = 5   # n is the word length\n\nos.chdir(\'C:/Users/Miles/Documents/Brown/Ogbunu Lab/Evo Word\')   # sets the cwd as C:/.../Evo Word\n\n\nwith open(\'%s_Letter_OneGrams_1900_2000.csv\'%str(n), \'r\') as f:\n    reader = csv.reader(f)\n    data = list(reader)                        # googles ngram data\n    \n\nfor i in range(1,len(data)):\n    data[i][1:] = map(int,data[i][1:])         # the above makes all elements str so this turns counts to int\n    \n\nngram_words = []  \nfor i in range(1,len(data)):\n    ngram_words.append(data[i][0])             # words from googles ngram\n     \n\n\npop_cutoff = [201,1501,5001]    # these will be the popularity cutoffs used for word-lengths 3, 4, and 5 resp.\n\n# trimming down to popular words\npop_words = [data[i][0] for i in range(1,pop_cutoff[n-3])]   # most popular words according to the cutoffs \n\n# word constructor\ndef like_words(X,letter):                      # gives all words (in pop_words) off by the given letter (=int giving position)\n    Y = X[:letter]+X[letter+1:]\n    sims = []\n    for each in pop_words:\n        test = each[:letter]+each[letter+1:]\n        if test == Y:\n            sims.append(each)\n    sims.remove(X)\n    return sims\n\ndef like_words_in_set(X,letter,Set):            # gives all words (in set) off by the given letter (=int giving position)\n    Y = X[:letter]+X[letter+1:]\n    sims = []\n    for each in Set:\n        test = each[:letter]+each[letter+1:]\n        if test == Y:\n            sims.append(each)\n    sims.remove(X)\n    return sims\n\ndef new_word(X,letter):                         # picks a random word from like_words (if no like word then returns 0)\n    test_words = like_words(X,letter)\n    if len(test_words)>0:\n        new = np.random.choice(test_words)\n        return new\n    else:\n        return 0\n    \ndef all_like_words(X):                          # gives a list of all words one letter removed from the input (in pop_words)\n    all_words = [] \n    for j in range(len(X)):\n        all_words.extend(like_words(X,j))\n    return all_words    \n\n    \ndef all_like_words_in_set(X,Set):                   # gives a list of all words one letter removed from the input (in given set)\n    all_words = [] \n    for j in range(len(X)):\n        all_words.extend(like_words_in_set(X,j,Set))\n    return all_words    \n\n\nN = 50\ni = 0\nDa_Words = []\n\n# this section identifies N word pairs of words between which paths exist\n#-------------------------------------------------------------------------------------------------------------#\n# while i < N:\n#     condition = 0                          # initializes a break condition in the second while loop\n#     rnge = range(n)                         # for 3-mers only \n#     word1 = np.random.choice(pop_words)  \n#     Da_Words.append([word1])\n#     r = np.random.choice(rnge)\n#     rnge.remove(r)\n#     word2 = new_word(word1,r)\n#     i = i + 1\n#     while condition==0:\n#         if word2 == 0:\n#             Da_Words.remove(Da_Words[i-1])\n#             condition = 1                  # breaks the while loop if can\'t find a new_word\n#             i = i - 1\n#             continue\n#         if len(rnge)==0:\n#             Da_Words[i-1].append(word2)\n#             condition = 1                  # breaks the loop if it reaches the last letter\n#         else:\n#             r = np.random.choice(rnge)\n#             rnge.remove(r)\n#             word2 = new_word(word2,r)\n#-------------------------------------------------------------------------------------------------------------#\n\nDa_Words = [[\'DARTS\',\'FILED\'],[\'SHARP\',\'ATONE\']]   # can manually select the words to examine\n        \n# word chains\nda_words = Da_Words            # establishes a list of pairs of words (constructed from the scrabble library)\n\n\nbits = []\nfor i in range(2**n):                    # constructs the n-bit strings from 0 to (2^n)\n    bits.append((\'{0:0%sb}\'%n).format(i))\n\n\nindices = []\n\nfor each in bits:\n    b = list(each)\n    indices.append([n for (n, e) in enumerate(b) if e == \'1\'])\n    \n\nword_chain = []\n\nfor j in range(len(da_words)):\n    wordA = da_words[j][0]\n    wordB = da_words[j][1]\n    word_chain.append([wordA])\n    i = 1\n    while i<len(bits):\n        word = list(wordA)\n        for each in indices[i]:\n            word[each] = wordB[each]\n        word = \'\'.join(word)\n        word_chain[j].append(word)\n        i = i + 1\n        \n# Array of fitness values\n# time ~ 2.5 mins\nW = []\nfor i in range(len(Da_Words)):                        # this indexes over the pairs of words\n    W.append([])\n    for j in range(len(data[i][1:])):                 # this indexes over the years\n        W[i].append([])\n        for those in word_chain[i]:                   # this indexes over those in the word chain\n            if those in ngram_words:\n                index = 1 + ngram_words.index(those)  # we add 1 bc ngram_words is an index below words_from_google\n                W[i][j].append(data[index][j+1])\n            else:\n                W[i][j].append(0)\n      \n\n    \n# the word dictionary-graph\n# time ~ 300 ms\nword_graph_dict = {}\nfor each in pop_words:\n    word_graph_dict.update({each:all_like_words(each)})\n    \n\n# defining the graph class  \n\n#----------------------------------------------------------------------------------#\nclass Graph(object):\n\n    def __init__(self, graph_dict=None):\n        """ initializes a graph object \n            If no dictionary or None is given, \n            an empty dictionary will be used\n        """\n        if graph_dict == None:\n            graph_dict = {}\n        self.__graph_dict = graph_dict\n\n    def vertices(self):\n        """ returns the vertices of a graph """\n        return list(self.__graph_dict.keys())\n\n    def edges(self):\n        """ returns the edges of a graph """\n        return self.__generate_edges()\n\n    def add_vertex(self, vertex):\n        """ If the vertex "vertex" is not in \n            self.__graph_dict, a key "vertex" with an empty\n            list as a value is added to the dictionary. \n            Otherwise nothing has to be done. \n        """\n        if vertex not in self.__graph_dict:\n            self.__graph_dict[vertex] = []\n\n    def add_edge(self, edge):\n        """ assumes that edge is of type set, tuple or list; \n            between two vertices can be multiple edges! \n        """\n        edge = set(edge)\n        (vertex1, vertex2) = tuple(edge)\n        if vertex1 in self.__graph_dict:\n            self.__graph_dict[vertex1].append(vertex2)\n        else:\n            self.__graph_dict[vertex1] = [vertex2]\n\n    def __generate_edges(self):\n        """ A static method generating the edges of the \n            graph "graph". Edges are represented as sets \n            with one (a loop back to the vertex) or two \n            vertices \n        """\n        edges = []\n        for vertex in self.__graph_dict:\n            for neighbour in self.__graph_dict[vertex]:\n                if {neighbour, vertex} not in edges:\n                    edges.append({vertex, neighbour})\n        return edges\n\n    def __str__(self):\n        res = "vertices: "\n        for k in self.__graph_dict:\n            res += str(k) + " "\n        res += "\\nedges: "\n        for edge in self.__generate_edges():\n            res += str(edge) + " "\n        return res\n    \n    def find_path(self, start_vertex, end_vertex, path=None):\n        """ find a path from start_vertex to end_vertex \n            in graph """\n        if path == None:\n            path = []\n        graph = self.__graph_dict\n        path = path + [start_vertex]\n        if start_vertex == end_vertex:\n            return path\n        if start_vertex not in graph:\n            return None\n        for vertex in graph[start_vertex]:\n            if vertex not in path:\n                extended_path = self.find_path(vertex, \n                                               end_vertex, \n                                               path)\n                if extended_path: \n                    return extended_path\n        return None\n\n    def find_all_paths(self, start_vertex, end_vertex, path=[]):\n        """ find all paths from start_vertex to \n            end_vertex in graph """\n        graph = self.__graph_dict \n        path = path + [start_vertex]\n        if start_vertex == end_vertex:\n            return [path]\n        if start_vertex not in graph:\n            return []\n        paths = []\n        for vertex in graph[start_vertex]:\n            if vertex not in path:\n                extended_paths = self.find_all_paths(vertex, \n                                                     end_vertex, \n                                                     path)\n                for p in extended_paths: \n                    paths.append(p)\n        return paths\n    \n    def vertex_degree(self, vertex):\n        """ The degree of a vertex is the number of edges connecting\n            it, i.e. the number of adjacent vertices. Loops are counted \n            double, i.e. every occurence of vertex in the list \n            of adjacent vertices. """ \n        adj_vertices =  self.__graph_dict[vertex]\n        degree = len(adj_vertices) + adj_vertices.count(vertex)\n        return degree\n    \n    def density(self):\n        """ method to calculate the density of a graph """\n        g = self.__graph_dict\n        V = len(g.keys())\n        E = len(self.edges())\n        return 2.0 * E / (V *(V - 1))\n    \n    def is_connected(self, \n                     vertices_encountered = None, \n                     start_vertex=None):\n        """ determines if the graph is connected """\n        if vertices_encountered is None:\n            vertices_encountered = set()\n        gdict = self.__graph_dict        \n        vertices = list(gdict.keys()) # "list" necessary in Python 3 \n        if not start_vertex:\n            # chosse a vertex from graph as a starting point\n            start_vertex = vertices[0]\n        vertices_encountered.add(start_vertex)\n        if len(vertices_encountered) != len(vertices):\n            for vertex in gdict[start_vertex]:\n                if vertex not in vertices_encountered:\n                    if self.is_connected(vertices_encountered, vertex):\n                        return True\n        else:\n            return True\n        return False\n    \n#----------------------------------------------------------------------------------#\n    \n# word paths in the popular set\n# time ~ 400 ms\npaths = []\nfor k in range(len(word_chain)):\n    wds_in_pop = [w for w in word_chain[k] if w in pop_words]\n    wds_in_pop_dict = {}\n    for each in wds_in_pop:\n        in_graph = [wds for wds in all_like_words(each) if wds in wds_in_pop]\n        wds_in_pop_dict.update({each:in_graph})\n    wds_in_pop_grph = Graph(wds_in_pop_dict)\n    paths.append(wds_in_pop_grph.find_all_paths(word_chain[k][0],word_chain[k][-1]))\n    \npaths_dict = {}\nfor i in range(len(paths)):\n    for j in range(len(paths[i])):\n        paths_dict.update({(i,j):paths[i][j]})\n\n# turning the list of lists into a dictionary keyed by tuples (paths will have the form {(i,j):path})\n\npaths = paths_dict\n\n\n# uphill or not for each path and year\n# time ~ 310 ms\ndef is_uphill(self,year):      # checks if a path (self) is uphill in a given year\n    yr = data[0].index(year)\n    a = 0\n    k = ngram_words.index(self[0])\n    i = 0\n    condition = 0\n    while condition==0:\n        if data[k+1][yr]<a:\n            is_up = 0\n            condition = 1\n        elif i==len(self):\n            is_up = 1\n            condition = 1\n        else:\n            a = data[k+1][yr]\n            k = ngram_words.index(self[i])\n            i = i + 1\n            \n    return is_up\n\n\npaths_count = []                             # gives the lengths of the paths \nfor tupl in paths:\n    paths_count.append(len(paths[tupl]))\n    \ncheck = {}                                   # checks which paths are uphill and which are over all paths and all years\nfor tupl in paths:\n    check.update({tupl:[]})\n    path = paths[tupl]\n    for each in data[0][1:]:\n        check[tupl].append(is_uphill(path,each))\n            \nchecks_count = []                            # counts how often paths are uphill\nfor tupl in check:\n    checks_count.append(len(check[tupl]))\n\nuphills = []                                 # appends to \'uphills\' the paths with more than 20 uphill years\nfor tupl in check:\n    to_sum = [nums for nums in check[tupl] if nums==1]\n    if sum(to_sum)>-10:     # this cutoff can be changed to the desired constraint\n        uphills.append(tupl)\n\n# uphill paths\n# gives a list of mostly uphill paths (mostly means more than 10 uphill years)\nuphill_words = []\nfor each in uphills:\n    uphill_words.append(paths[each])\n    print(paths[each])\n    \n# identifies the unique pair of uphill words    \nup_pairs = list(set([(path[0],path[-1]) for path in uphill_words]))   ')


# ### List of Mostly Uphill Paths

# In[41]:


# gives a list of mostly uphill paths (mostly means more than 10 uphill years)
uphill_words = []
for each in uphills:
    uphill_words.append(paths[each])
    print(paths[each])


# ### Plotting the $C_W$ Over Time in the Uphill Words

# In[57]:


import matplotlib.ticker as ticker

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

def path_time(path,year):           # calculates Cw for a given path and year
    time = 0
    for i in range(len(path)-1):
        yr = data[0].index(year)
        wrda = 1+ngram_words.index(path[i])
        wrdb = 1+ngram_words.index(path[i+1])
        df = data[wrdb][yr]-data[wrda][yr]
        time = time + 1/df
    return time

Cw = {}   # Cw will have the form {(i,j):[list 1,list 2]} where list 1 is Cw for each yr and list 2 is check[i][j]

for tup in uphills:
    pth = paths[tup]
    Cw.update({tup:[check[tup],[]]})
    temp = []
    text = '\n'.join(pth)                    # gives a formatted path (to be placed next to graph)
    Path = ' '.join(pth)
    for each in data[0][1:]:
        temp.append(path_time(pth,each))  
        Cw[tup][1].append(path_time(pth,each))  
        
    # identifies the years where there was an uphill trajectory
    up_yrs = [data[0][1:][x] for x in range(len(data[0][1:])) if check[tup][x]==1]

    # the plot
    fig,ax = p.subplots(figsize=[10,6])
    ax.plot(data[0][1:],temp)
    ax.plot(up_yrs,np.zeros(len(up_yrs)),'o')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    right = ax.get_xlim()[1]
    upper = sum(ax.get_ylim())
    p.text(1+right,upper/2,text, fontsize=15,horizontalalignment='left',verticalalignment='center')
    p.xlabel('Year',fontsize=20)
    p.ylabel('$C_W$',fontsize=20)
    p.xticks(rotation=75)
    p.tick_params(labelsize=15,axis='x',direction='in',top=1,right=1)
    p.tick_params(labelsize=15,axis='y',direction='in',top=1,right=1)
    p.title('%s to %s $C_W$' %(pth[0],pth[-1]),fontsize=20)
    
#     p.savefig('C:/Users/Miles/Documents/Brown/Ogbunu Lab/Evo Word/EvoSpeed/Figures/Uphill 5 Letter Words/%s Cw' %Path)

    
# saving Cw as a CSV
for tupl in uphills:
    x = list(map(int,data[0][1:]))
    y = Cw[tupl][0]
    z = Cw[tupl][1]
    string = ' '.join(paths[tupl])
    np.savetxt('C:/Users/Miles/Documents/Brown/Ogbunu Lab/Evo Word/EvoSpeed/Cw/English/%s Letter/Cw %s.csv' %(n,string),(x,y,z),delimiter=',')


# ### Defining the Epistasis Arrays

# In[58]:


from scipy.linalg import hadamard
import matplotlib.ticker as ticker
import matplotlib.cm

year = data[0][1:]              # string of the years
order = ['$0^{th}$','$1^{st}$','$2^{nd}$','$3^{rd}$','$4^{th}$','$5^{th}$'][0:n+1]   # labels for the orders

H = hadamard(len(bits))         # Hadamard matrix of the appropriate size
         
v = np.array([[0.5,0],[0,-1]])     # this initializes the diagonal matrix V_1
V = v


for i in range(n-1):   # this defines the diagonal matrix V: range number should be one less than the word length 
    V = np.kron(V,v)

M = np.matmul(V,H)     # here we multiply the diagonal matrix V with H
E = []

for j in range(len(W)):
    E.append([])
    for i in range(len(year)):
        E[j].append(np.array(M.dot(W[j][i])))

# definfing the mean absolute epistasis arrays

eabs = []
orders = list(map(len,indices))   # this gives a list of the order (weight) of each bit

pascal = [[1,3,3,1],[1,4,6,4,1],[1,5,10,10,5,1]][n-3] # this gives the multiplicities of the orders for a given n (nth row of the Pascal triangle)

# this constructs the eabs array: a collection of averages of absolute values of elements in E (for each year and word chain)

empty = []                    # this will house the orders and is constructed to structure the eabs list
for r in range(n+1):
    empty.append([])

for k in range(len(W)):
    eabs.append(empty)        # this gives eabs n+1 bins to house the n+1 orders
    for each in E[k]:
        s = sum(list(map(abs,each)))              # delete the line below to turn on normalization
#         s = 1                                   # this shuts off the normalization of the epistasis
        for o in range(len(order)):
            epi_sum = sum(list(map(abs,[each[n] for (n,i) in enumerate(orders) if i==o]))) # sum of the abs(orders)
            eabs[k][o].append(epi_sum/(pascal[o] * s))

            
# the epistasis disaggregator

e_disag = []
for j in range(len(W)):
    e_disag.append([])
    for o in range(len(order)):
        e_disag[j].append([])
        index = [n for (n,i) in enumerate(orders) if i==o]
        for x in range(len(index)):
            e_disag[j][o].append([])
            for each in E[j]:
                s = sum(list(map(abs,each)))   
                e_disag[j][o][x].append(abs(each[index[x]])/s)
     
    
E = np.array(E)  # making E into an array
                
# colors

colors = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

cmap = matplotlib.cm.get_cmap('Purples')

rgba1 = cmap(0.5)
rgba2 = cmap(0.6)


# ### Storing Epistasis into CSV files

# In[59]:


import os
import csv


os.chdir('C:/Users/Miles/Documents/Brown/Ogbunu Lab/Evo Word')   # sets the cwd as C:/.../Evo Word

# gives a list of mostly uphill paths (mostly means more than 10 uphill years)
uphill_words = []
for each in uphills:
    uphill_words.append(paths[each])
    print(paths[each])
    
# identifies the unique pair of uphill words    
up_pairs = list(set([(path[0],path[-1]) for path in uphill_words]))   


# organizing the fitness values into a csv
for pair in up_pairs:    # can use "up_pairs" here in place of Da_Words if want just the uphill words
    fitness = []
    j = Da_Words.index([pair[0],pair[1]])
    for word in word_chain[j]:
        if word in ngram_words:
            k = 1 + ngram_words.index(word)
            fitness.append(data[k][1:])
        else:
            fitness.append(list(map(int,np.zeros(len(data[0])-1))))
            
    np.savetxt('Fitness/English/%s Letter/fitness %s to %s.csv' %(str(n),pair[0],pair[1]),[],delimiter=',')
    
    with open('Fitness/English/%s Letter/fitness %s to %s.csv' %(str(n),pair[0],pair[1]),'w') as file:
        writer = csv.writer(file,delimiter=',')
        for line in fitness:
            writer.writerow(line)


# organizing the epistasis values into a csv
for pair in up_pairs:    # can use "up_pairs" here in place of Da_Words if want just the uphill words
    wrdA = pair[0]
    wrdB = pair[1]
    k = Da_Words.index([wrdA,wrdB])
    np.savetxt('Epistasis/Epistasis Data/English/%s Letters/Epistasis %s to %s.csv' %(str(n),wrdA,wrdB),[],delimiter=',')
    with open('Epistasis/Epistasis Data/English/%s Letters/Epistasis %s to %s.csv' %(str(n),wrdA,wrdB),'w') as file:
        writer = csv.writer(file,delimiter=',')
        writer.writerow(word_chain[k])
        for i in range(len(data[0][1:])):
            writer.writerow(E[k,i,:])
        writer.writerow([])
        


# ### Plotting the Epistasis

# In[ ]:


import numpy as np
from scipy.integrate import odeint
from scipy.linalg import hadamard
import pylab as p
import os
import matplotlib.cm

# disaggregated

order_index = [[n for (n,i) in enumerate(orders) if i==o] for o in range(len(order))]

zeroth = [bits[i] for i in order_index[0]]
first = [bits[i] for i in order_index[1]]
second = [bits[i] for i in order_index[2]]
third = [bits[i] for i in order_index[3]]
fourth = [bits[i] for i in order_index[4]]
fifth = [bits[i] for i in order_index[5]]    # for 5-mers


# labels = [zeroth,first,second,third,fourth]
labels = [zeroth,first,second,third,fourth,fifth]   # for 5-mers


for pair in up_pairs:
    
    word_index = Da_Words.index(list(pair))
    
    fig,ax = p.subplots(figsize=[15,10])
    
    for j in range(len(order)):

        cmap = matplotlib.cm.get_cmap(colors[j])
#         print(colors[j])
        color = np.linspace(0.5,1,len(order_index[j]))

        for i in range(len(order_index[j])):
            ax.plot(year,e_disag[word_index][j][i],label=labels[j][i],color=cmap(color[i])[:-1])

    p.xlabel('Year',fontsize=20)
    p.ylabel('Epistasis (Absolute Value)',fontsize=20)
    p.xticks(rotation=75)
    p.tick_params(labelsize=15,axis='x',direction='in',top=1,right=1)
    p.tick_params(labelsize=15,axis='y',direction='in',top=1,right=1)
    p.title('Epistasis in %s to %s (Absolute Value)' %(word_chain[word_index][0],word_chain[word_index][-1]),fontsize=30)
    p.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=15)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    p.tight_layout()

    p.savefig('C:/Users/Miles/Documents/Brown/Ogbunu Lab/Evo Word/Epistasis/Figures/Uphill Disaggregated/English/5 Letter/%s to %s disaggregated' %(word_chain[word_index][0],word_chain[word_index][-1]))

    
# aggregated

for pair in up_pairs:
    
    j = Da_Words.index(list(pair))
    
    fig,ax = p.subplots(figsize=[10,6])
    
    for i in range(len(order)):
        p.plot(year,eabs[j][i],label=order[i])

    p.xlabel('Year',fontsize=20)
    p.ylabel('Epistasis (Mean Absolute Value)',fontsize=20)
    p.xticks(rotation=75)
    p.tick_params(labelsize=15,axis='x',direction='in',top=1,right=1)
    p.tick_params(labelsize=15,axis='y',direction='in',top=1,right=1)
    p.title('Epistasis (Absolute Mean) in %s to %s' %(da_words[j][0],da_words[j][1]),fontsize=30)
    p.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=15)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    p.tight_layout()

    p.savefig('C:/Users/Miles/Documents/Brown/Ogbunu Lab/Evo Word/Epistasis/Figures/Uphill Epistasis/English/5 Letter/%s to %s (normalized)' %(da_words[j][0],da_words[j][1]))

