# Part-of-Speech Tagger 

**For my code and more details, please see my notebook [here](https://github.com/HeleneFabia/pos-tagger/blob/main/postagger.ipynb)**

### Data
The [CoNLL dataset](https://www.clips.uantwerpen.be/conll2000/chunking/) consists of roughly 10,000 sentences, with each word having a corresponding part-of-speech tag, like the following:

| Token     | POS |
|-----------|-----|
| When      | WRB |
| bank      | NN  |
| financing | NN  |
| for       | IN  |
| the       | DT  |
| buy-out   | NN  |
| collapsed | VBD |
| last      | JJ  |
| week      | NN  |
| ,         | ,   |
| so        | RB  |
| did       | VBD |
| UAL       | NNP |
| 's        | POS |
| stock     | NN  |
| .         | .   |

A complete look-up table for each part-of-speech tag can be found [here](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).

### Model Architecture

According to Wikipedia (I know, I know, quoting from Wikipedia...), "words that are assigned to the same part of speech generally display similar syntaxic behavior (they play similar roles within the grammatical structure of sentences)". This means that the POS of a word depends on its role in the current sentence. Consider the word "right" in the following two sentences: "This is the *right* (JJ) answer" vs. "You have the *right* (NN) to remain silent"). By itself, you could not assign the correct part of speech to it, but only with the help of the rest of sentence. Hence, we want to use whole sequences as a model's input, not just individual words. I decided to go for a very simple approach and use a vanilla RNN as my sequence model.

An overview of the complete architecture I used can be seen here:

![model](https://github.com/HeleneFabia/pos-tagger/blob/main/images/model_overview.png)

One might wonder what a convolutional layer is doing in this architecture. Well, if you have a look at the output of the RNN model, you see that the shape of this output is of size `(sequence_length, hidden_size)`. The final output I need should be of size `(sequence_length, num_classes)`, so that I end up with a probability for each step in the sequence and for POS tag. This computation cannot done by a standard dense layer, since it would squeeze the dimension regarding sequence length. In Keras, there's what is called Time-Distributed Dence (TDD), which is essentially a dense layer but considering time steps. Unfortunately, there's no implementation of this kind of layer in PyTorch. However, a possible subsitution for it can be a convultional layer. One important characteristic of a Time-Distributed Dense is that it applies the same weights at each time step – just like a convolutional layer does with the help of its kernel. Below, a more detailed visualization of what the convolutional layer does.

![conv2d](https://github.com/HeleneFabia/pos-tagger/blob/main/images/conv.png)

### Results

After training for around 40 epochs, it achieved 0.8890 and 0.8667 top-1 accuracy on the validation and test set, respectively. The top-1 accuracy for each class looks like this:

![accuracy](https://github.com/HeleneFabia/pos-tagger/blob/main/images/accuracy.png)

An in-depth written-up analysis as well as further ideas for improvements can be found in the last sections of my [notebook](https://github.com/HeleneFabia/pos-tagger/blob/main/postagger.ipynb)

