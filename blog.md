# An Introductory Guide to Finetuning Large Language Models

### What is a Large Language Model?

The term "Large Language Model" can be broken down into two parts.

A "Language Model" is a machine learning model that is trained to perform well on tasks related to text/language such as Classification, Summarization, Translation, Prediction and Generation. With recent developments in this field, one can engage in long-form conversations, any-to-any language translation, sentiment and data analysis, question answering, etc on demand and for free! The potential use-cases for Language Models are virtually unlimited, and as technology continues to evolve and as more research is done, we can expect even more innovative applications to emerge.

The "Large" part signifies that LLMs are a kind of deep learning algorithm. They have millions to billions (and now even trillions, as in the case of GPT4) of trainable parameters. This massive amount of parameters enables LLMs to achieve exceptional performance across a wide range of tasks.

Today, there is a growing number of LLMs that are gaining recognition, including GPT (most notably, ChatGPT), Codex, BLOOM, BERT, LLaMA, LaMDA, and PaLM. To enable the development and training of these complex models, large technology companies are leading the way as they possess the infrastructure and expertise. These companies possess the know-how and compute to train and fine-tune models, allowing them to achieve impressive results across a wide range of natural language processing tasks.

### History of Large Language Models

LLMs have had significant evolution since their origin. The first models were shallow and relied on the n-gram technique that simply counted occurrences of words or phrases. Later on, neural language models came about, such as recurrent neural networks (RNNs, LSTMs and GRUs) that enabled more complex models to rise.

One of the most significant advancements in LLMs came about in 2017, where the "Transformer" architechture was introduced in the paper titled "Attention Is All You Need". It enabled the training of deep neural nets to become highly parallelizable as compared to previous techniques. The attenion mechanism that was introduced not only enabled the development of LLMs with billions of parameters but also found its way into multiple other areas. With the ongoing R&D in this area, it is likely that models will continue to become even more powerful and versatile in the future.

BERT, which is one such model based on the transformer architecture, is what we will be looking at for demonstration and finetuning in this blog.

### What is Finetuning?

Finetuning is the process of taking a pre-trained language model and further training it on a specific task, such as sentiment analysis, to improve its performance on that particular task. This is achieved by taking the pre-trained model and training it on a smaller dataset that is specific to the task. It allows the model to adapt its existing knowledge rather than having to learn everything from scratch.

In this blog, we will be looking at finetuning BERT (a language model developed by Google), that has been pre-trained on a large corpus of text allowing it to capture a wide range of language patterns and contextual relationships. The task that we will be finetuning it on is Masked Language Modelling.

### What is Masked Language Modelling?

MLM is a task which involves masking/removing (think of it like adding blanks to a sentence to create fill-in-the-blanks style questions) some of the words in a sentence and making a model predict those words based on remaining contextual information. In BERT, this is used as a pre-training task to allow it to learn a general understanding of the language. Masking involves randomly or algorithmically replacing words with a special `[MASK]` token.

An example of masking is as follows:

**Original Text**: There are so many discoveries and advancements in the field of AI and ML these days. What a time to be alive!

**After Masking**: There are so many [MASK] and advancements in the field of AI and [MASK] these days. What a time to be [MASK]!

**Probable Mask Prediction**: There are so many [innovations] and advancements in the field of AI and [technology] these days. What a time to be [alive]!

### Getting started with BERT

Access to models like BERT has been made very easy over the years, thanks in large part to platforms like Hugging Face, which provides a user-friendly interface for accessing and utilizing pre-trained language models. They also offer a wide range of pre-trained models and tools for finetuning them on specific tasks. With Hugging Face, even those without a deep understanding of machine learning can easily use pre-trained models like BERT to improve the performance of their NLP applications.

Let's dive into some code and get our hands dirty with BERT. The prerequisites for this is that you need a working installation of Python.

- Installing the HuggingFace Transformers Library

```
pip install transformers
```

- Create a Python file or Jupyter notebook with the following code:

```python
import json
from transformers import pipeline

masked_sentence = "There are so many [MASK] and advancements in the field of AI and [MASK] these days. What a time to be [MASK]!"
fill_masker = pipeline(model="distilbert-base-uncased")
print(json.dumps(fill_masker(masked_sentence), indent=2))
```

Instead of using BERT, the above code uses DistilBERT - a smaller, faster and almost equally performant model. The pipeline invokes the model and retrieves the top 5 (can be changed by passing the `top_k` attribute which defaults to 5) predictions for each mask. Since there are three masks, the model will return a total of 15 predictions (5 for each word). The output is as follows:

<details>
<summary>Output</summary>
<br />

```json
[
  [
    {
      "score": 0.16455209255218506,
      "token": 15463,
      "token_str": "innovations",
      "sequence": "[CLS] there are so many innovations and advancements in the field of ai and [MASK] these days. what a time to be [MASK]! [SEP]"
    },
    <TRUNCATED>
  ],
  [
    {
      "score": 0.02868303284049034,
      "token": 2974,
      "token_str": "technology",
      "sequence": "[CLS] there are so many [MASK] and advancements in the field of ai and technology these days. what a time to be [MASK]! [SEP]"
    },
    <TRUNCATED>
  ],
  [
    {
      "score": 0.023737657815217972,
      "token": 5541,
      "token_str": "creative",
      "sequence": "[CLS] there are so many [MASK] and advancements in the field of ai and [MASK] these days. what a time to be creative! [SEP]"
    },
    <TRUNCATED>
  ]
]
```

</details>

As can be seen from the output, the model assigns a confidence "score" for different predictions. It then sorts them based on this score and higher score predictions show up in the top_k outputs.

### Finetuning Task

Since we'd like to finetune BERT, we need to define the task at which we want to make it better. Our goal is to bias the masked word predictions to have more positive sentiment in the context of product reviews at Nike. We can do so by providing the model some labeled data to retrain on, where each sentence has a corresponding positive or negative sentiment.

In simpler words, let's say we have the sentence "Nike shoes are very [MASK]". The predictions that we want for the masked token here are "popular", "durable", "comfortable" as opposed to "expensive", "ugly", "heavy".

For this task, we need a dataset that contains sentence examples of the above format. We could use an online dataset but for the sake of the example, let's synthesize our own. We can do the synthesis using the code below. Basically, we define some hardcoded sentence formats and use it to generate sentences by filling placeholder values like adjectives, products, etc. Note that some sentences created this way will not make complete sense but it doesn't matter much in this task, so long as we are providing the model with enough information about what we are trying to do.

<details>
<summary>Code</summary>
<br />

```python
PRODUCTS = [
    'gym wear', 'jackets', 'shirts',
    'running shoes', 'basketballs', 'caps', 'pants', 'socks',
    'trousers', 'training shoes', 'basketball shoes', 'shoes',
    'athletic wear', 'sports wear', 'footballs',
    'performance gear', 'hats', 'sweaters', 'tshirts', 'wristbands',
    'backpacks', 'tshirts', 'hoodies', 'trainers',
    'soccer shoes',
]

POSITIVE_SENTIMENT_ADJECTIVES = [
    'user-friendly', 'innovative', 'support', 'good-looking', 'efficient',
    'stylish', 'breathable', 'flexibility', 'trendsetting', 'performance',
    'impressive', 'resilient', 'durability', 'durable', 'athletic', 'breathability',
    'cheap', 'comfort', 'comfortable', 'inexpensive', 'premium', 'sleek',
    'performance-oriented', 'fashionable', 'quality', 'flexible', 'stability',
    'look', 'functional', 'sporty', 'lightweight', 'bounce', 'grip', 'modern',
    'fit', 'ergonomic', 'versatile', 'style', 'design', 'cushioning', 'traction',
    'high-quality', 'revolutionary'
]

NEGATIVE_SENTIMENT_ADJECTIVES = [
    'uncomfortable', 'flimsy', 'poor quality', 'outdated', 'unfashionable',
    'heavy', 'inferior', 'unathletic', 'expensive', 'costly',
    'overpriced', 'defective', 'ugly', 'dirty', 'faulty',
    'non-durable', 'tacky', 'lacking in performance', 'clunky', 'bulky',
    'awkward', 'disappointing', 'unreliable', 'displeasing', 'unsatisfactory'
]

ADJECTIVES = POSITIVE_SENTIMENT_ADJECTIVES + NEGATIVE_SENTIMENT_ADJECTIVES

COMPANIES = [
    # repeat a couple of times for higher positive examples of Nike
    'nike', 'nike', 'nike', 'nike', 'nike', 'nike', 'nike', 'adidas', 'puma',
    'under armour', 'reebok', 'converse', 'vans', 'fila', 'asics'
]

JOINERS = [
    'are', 'is', 'offer', 'provide', 'feature', 'boast',
    'are known for being', 'are recognized for being', 'are famous for being',
    'are renowned for being', 'are praised for being',
]

def create_sample_dataset(dataset_size):
    # We will also add some null values to the dataset to demonstrate
    # the Data Integrity capabilities of UpTrain
    nullify_ratio = 0.05
    nullify_count = int(nullify_ratio * dataset_size)
    data = {
        "version": "0.1.0",
        "source": "sample",
        "url": "self-generated",
        "data": []
    }
    sentences = []

    for _ in range(dataset_size):
        company = random.choice(COMPANIES)
        joiner = random.choice(JOINERS)
        product = random.choice(PRODUCTS)
        label = random.randint(0, 3)

        # We bias the positive sentiment data to have a higher ratio
        if label == 0:
            adjective = random.choice(NEGATIVE_SENTIMENT_ADJECTIVES)
        else:
            label = 1
            adjective = random.choice(POSITIVE_SENTIMENT_ADJECTIVES)

        # Additionally, you could expand on list of possible sentences
        # or use a combination of real-life datasets
        if random.randint(0, 1) == 0:
            sentence = f'{company} {product} {joiner} {adjective}'
        else:
            sentence = f'{product} made by {company} {joiner} {adjective}'
        
        sentences.append({ "text": sentence, "label": label })
    
    # Make some values null to make sure UpTrain data integrity check is working
    for _ in range(nullify_count):
        element = random.choice(sentences)
        element['text'] = None
    
    data["data"] = sentences
    return data
```

</details>

Now that we have our dataset, we could go ahead and try finetuning our model but it will not really improve the performance much. The model will be fed both positive and negative sentiment data and it may not learn to prioritize the positive sentiment predictions as expected in the task. This is where UpTrain comes in - with just a few lines of code, one can define a "Signal" which can be used as a filter for the dataset. It has other use cases as well, which you can find by checking out the UpTrain repository. Let's take a look at how to use UpTrain signals.

In the code below, we define three functions. These functions are callbacks that UpTrain signals will use to determine whether some data from our dataset is relevant to our finetuning task at hand. We can chain multiple signals as well as mix and match them by using simply `&` (combining signals logically with the AND operator) and `|` (combining signals logically with the OR operator) operators.

<details>
<summary>Code</summary>
<br />

```python
def nike_text_present_func (inputs, outputs, gts=None, extra_args={}):
    """Checks if the word "Nike" is present in the text or not"""
    is_present = []
    for text in inputs["text"]:
        present = False
        if text is not None:
            text = text.lower()
            present = bool("nike" in text)
        is_present.append(present)
    return is_present

def nike_product_keyword_func (inputs, outputs, gts=None, extra_args={}):
    """Checks if the sentence contains a product associated with Nike or not"""

    is_present = []
    for text in inputs["text"]:
        present = False
        if text is not None:
            text = text.lower()
            present = any(word in text for word in PRODUCTS)
        is_present.append(present)
    return is_present

def is_positive_sentiment_func (inputs, outputs, gts=None, extra_args={}):
    """Determines if an input sentence has a positive sentiment or not"""

    vader_sia = SentimentIntensityAnalyzer() # from nltk module
    is_positive = []
    for text in inputs["text"]:
        positive = False
        if text is not None:
            text = text.lower()
            if vader_sia.polarity_scores(text)["compound"] >= 0:
                positive = any(word in text for word in POSITIVE_SENTIMENT_ADJECTIVES)
        is_positive.append(positive)
    return is_positive

cfg = {
    'checks': [
        {
            "type": uptrain.Monitor.EDGE_CASE,
            "signal_formulae": \
                uptrain.Signal("Is 'Nike' text present?", nike_text_present_func) &
                uptrain.Signal("Is it a Nike product?", nike_product_keyword_func) &
                uptrain.Signal("Is positive sentiment?", is_positive_sentiment_func)
        },

        {
            "type": uptrain.Monitor.DATA_INTEGRITY,
            "measurable_args": {
                "type": uptrain.MeasurableType.INPUT_FEATURE,
                "feature_name": "text"
            },
            "integrity_type": "non_null"
        }
    ],

    # Define where to save the retraining dataset
    "retraining_folder": uptrain_save_fold_name,
    
    # Define when to retrain, define a large number because we
    # are not retraining yet
    "retrain_after": 10000000000,

    "logging_args": {"st_logging": True},
}

dashboard_name = "llm_bert_example"
framework = uptrain.Framework(cfg)
```

</details>

There are a few things to look at here. UpTrain provides different Monitors for monitoring performance, checking for data distribution shifts, and collecting edge cases to retrain upon, among other things. Here, we use the EDGE_CASE monitor and provide it with our signals. We also add a data integrity check to make sure that none of our data contains null values. All monitoring related activities will show up on UpTrain's live dashboard. Once processing of this part completes, we will have created a retraining dataset that contains examples that satisfy the requirements of the signals above. This dataset is not only smaller that the original (whether it be synthesized or obtained from a real source) but also contains only specific data that is relevant to the finetuning task.

Now that we have our retraining dataset that is specific to our finetuning task, we can begin retraining. HuggingFace provides APIs that make training and finetuning models really simple. To learn how we can do this for the example, checkout the source code for the entire example [here](https://github.com/uptrain-ai/uptrain/blob/main/examples/finetuning_LLM/).

In conclusion, finetuning large language models like BERT can be a powerful tool for solving specific natural language processing tasks. UpTrain provides a simple and easy to use interface that requires minimal code to perform model monitoring, data drift checks, data integrity checks, edge case detection, model bias tracking, custom metric monitoring and much more. Checkout the UpTrain repository [here](https://github.com/uptrain-ai/uptrain).
