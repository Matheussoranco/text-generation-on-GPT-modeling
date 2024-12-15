# text-generation-on-GPT-modeling
A Text generation GPT model. It implements an autoregressive language model using a miniature version of the GPT model. The model consists of a single Transformer block with causal masking in its attention layer. We use the text from the IMDB sentiment classification dataset for training and generate new movie reviews for a given prompt. When using this script with your own dataset, make sure it has at least 1 million words.

This repository includes the code and a notebook used to test and execute the code. This script should be run on a GPU.

Sources:

    Keras Text Generation with Miniature GPT
    Improving Language Understanding by Generative Adversarial Networks
    Language Models are Unsupervised Multitask Learners
    Exploring the Limits of Language Modeling
