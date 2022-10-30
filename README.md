<!--
**TreeCen/TreeCen** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->

# TreeCen:Building Tree Graph for Scalable Semantic Code Clone Detection
TreeCen is a scalable tree-based code clone detector, which satisfies scalability while detecting semantic clones effectively.
Given the source code, we first extract ASTs based on static analysis to transform them into simple graph representations (i.e., tree graph) according to the node type, rather than traditional AST matching.
Then we treat the tree graph as a network and adopt the centrality analysis to convert it into a fix-length vector.
By this, the final vector is only 72-dimensional but contains complete structural information of the AST.
Finally, these vectors are fed into the machine learning model to detect clones.

TreeCen consists of four main phases: AST Extraction, AST Abstraction, Feature Extraction, Clone Detection:

1. AST Extraction: This phase aims to extract the AST for a method based on static analysis, whose input is the source code, and the output is an AST.

2. AST Abstraction: This phase is designed to simplify the AST while preserving its structural information.
The input is an AST, and the output is a tree graph for the AST, where each node represents a node type in the AST, and each edge conveys an edge relationship in the AST.

3. Feature Extraction:
In this phase, we assign centrality to each node within the tree graph generated in the AST abstraction phase. 
The output is centrality vectors containing concrete features about the AST.

4. Clone Detection:
Given the centrality vector of a pair of codes, we concatenate them into one vector and then feed it into a machine learning model to train the detector after annotating the corresponding labels (clone or non-clone).


# Project Structure  
```shell  
TreeCen 
|-- get_centmatrix.py  // implement the first three phases: AST Extraction, AST Abstraction, and Feature Extraction
|-- classification.py  // implement the Classification phase  
```

### Step1: Get centrality matrices
get_centmatrix.py: The file is used to get six types of centrality measures of source code. 

```
python get_centmatrix.py
```

The input is a folder containing the source code files and the output is a Json file containing a dictionary. The content of the dictionary is the feature vector of the source code obtained by each centrality measures.

### Step2: Classification
classification.py: The file is used to train the clone detector and predict the clones using each of the seven machine learning algorithms. 

```
python classification.py
```

Input:

1. The Json file output by get_centmatrix.py.

2. The CSV file with source code labels (clone or non-clone).

Output is the prediction result.

# Publication
Yutao Hu, Deqing Zou, Junru Peng, Yueming Wu, Junjie Shan, and Hai Jin. 2022. TreeCen: Building Tree Graph for Scalable Semantic Code Clone Detection. In 37th IEEE/ACM International Conference on Automated Software Engineering (ASE '22), October 10–14, 2022, Rochester, MI, USA. ACM, New York, NY, USA 12 Pages. https://doi.org/10.1145/3551349.3556927

If you use our dataset or source code, please kindly cite our paper:

```
@INPROCEEDINGS{treecen,
  author={Hu, Yutao, and Zou, Deqing and Peng, Junru and Wu, Yueming and Shan, Junjie and Jin, Hai},
  booktitle={2022 IEEE/ACM 37th International Conference on Automated Software Engineering (ASE)}, 
  title={TreeCen: Building Tree Graph for Scalable Semantic Code Clone Detection}, 
  year={2022},
  doi={10.1145/3551349.3556927}}
```

