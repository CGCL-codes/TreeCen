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
Given the source code, we first extract ASTs based on static analysis to transform them into simple graph representations (\ie tree graph) according to the node type, rather than traditional AST matching.
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


The source code and dataset of TreeCen will be published here after the paper is accepted.
