# text-generation-on-GPT-modeling
A Text generation GPT model. It implements an autoregressive language model using a miniature version of the GPT model. The model consists of a single Transformer block with causal masking in its attention layer. We use the text from the IMDB sentiment classification dataset for training and generate new movie reviews for a given prompt. When using this script with your own dataset, make sure it has at least 1 million words.

This repository includes the code and a notebook used to test and execute the code. This script should be run on a GPU.

Sources:</n>
https://keras.io/examples/generative/text_generation_with_miniature_gpt/ </n>
https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035 </n>
https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe </n>
https://arxiv.org/abs/2005.14165 </n>
