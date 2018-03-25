Potential directions/improvements
---------------------------------

- Question preprocessing to unify aliases - see kaggle-quora-dup submission
    - To streamline model learning

- Frequent words (>100?) collected, rest considered rare, replace with "yuzhshen"
    - For better generalization, prevent overfitting to rare words

- Different forms of preprocessing
    - kaggle-quora-dup method is fairly heavy-handed
    - potentially direct tokenization

- Remove questions with numbers? May overfit to numerical questions.

- Use pre-trained GloVe

- [Attention models](https://towardsdatascience.com/convolutional-attention-model-for-natural-language-inference-a754834c0d83)

Other resources
---------------

- [Siamese NN example](https://github.com/NVIDIA/keras/blob/master/examples/mnist_siamese_graph.py)