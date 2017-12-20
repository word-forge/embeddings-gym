# embeddings-gym

**It's under active development, currently unusable!**


## 1. Main Motivation

* OpenAI-Gym for embeddings
* Create a unified platform to develop, evaluate and compare embeddings (word embeddings or other).
* Be able to plug-in various implementations (Gensim, BigARTM, TensorFlow, PyTorch, etc).
* Be able to plug-in various datasets and evaluation techniques.
* Compare embeddings across multiple criteria in a unified and reproducible way.


## 2. High-level Components

* **Data generators** - objects that produce data batches for training and evaluation.
* **Models** - objects that have methods to train embeddings using the given data and to access the trained embeddings.
* **Evaluators** - objects that receive a data generator, a model and output a set of numbers - result of evaluation.
* **Utils** - Dictionary, Pipeline, Registry, etc.

### 2.1. Data Generators

**General use case:** receive path to data, generate data sample-by-sample, batch-by-batch or sentence-by-sentence.

**Classes:**

| Class | Input | One Sample |
| --- | --- | --- |
| Documents | Path to plain text files | ['sent1 word1', 'sent1 word2', 'sent2 word1' ] |
| Sentences | Path to plain text files | ['word1', 'word2'] |
| SlidingWindow | Documents or Sentences | { 'cur_word' : 'word1', 'neighbors' : ['word2', 'word3'] } |
| SlidingWindowAndGlobal | Documents or Sentences | { 'cur_word' : 'word1', 'neighbors' : ['word2', 'word3'], 'context' : ['sent1 word1', 'sent1 word2' ] } |
| Batcher | Any generator | Fetches a number of samples from nested generator, collates them and yields (like pytorch [default_collate](https://github.com/pytorch/pytorch/blob/bc6bd62bd6805136117d860819a361ef78608462/torch/utils/data/dataloader.py#L100)) }

### 2.2. Models

**General use cases:**
* train - receive generator and make embeddings
* use - pretty much like [gensim.models.KeyedVectors](https://radimrehurek.com/gensim/models/keyedvectors.html)
    * get list of known words
    * get vector by word
    * find similar words
    * save/load
    * export: get mapping { word : idx } and numpy.array of weights


**Classes:**
* GensimModel - Wraps an instance of gensim Word2Vec or FastText.
* TorchModel - Gets a torch.nn.Module, training args, optionally a Dictionary and fits the given nn.


### 2.3. Evaluators

**Concepts and their use cases:**
* Evaluator - a high-level evaluation algorithm that receives a set of models, a set of metrics and produces reports
* Metric - receives a model and optionally a data generator and output a set of numbers
    * WordSim - SimLex999-like evaluation
    * Analogies - GoogleAnalogies-like evaluation
    * TBD

## 3. Installation

TBD

## 4. Examples

TBD