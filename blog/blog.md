# An Introductory Guide to Finetuning Large Language Models

### What is a Large Language Model?

The term "Large Language Model" can be broken down into two parts.

A "Language Model" is a machine learning model that is trained to perform well on tasks related to text/language such as Classification, Summarization, Translation, Prediction and Generation. With recent developments in this field, one can engage in long-form conversations, any-to-any language translation, sentiment and data analysis, question answering, etc on demand and for free! The potential use-cases for Language Models are virtually unlimited, and as technology continues to evolve and as more research is done, we can expect even more innovative applications to emerge.

The "Large" part signifies that LLMs are a kind of deep learning algorithm. They have millions to billions (and now even trillions, as in the case of GPT4) of trainable parameters. This massive amount of parameters enables LLMs to achieve exceptional performance across a wide range of tasks.

Today, there is a growing number of LLMs that are gaining recognition, including GPT (most notably, ChatGPT), Codex, BLOOM, BERT, LLaMA, LaMDA, and PaLM. To enable the development and training of these complex models, large technology companies are leading the way as they possess the infrastructure and expertise. These companies possess the know-how and compute to train and fine-tune models, allowing them to achieve impressive results across a wide range of natural language processing tasks.

---

### History of Large Language Models

LLMs have had significant evolution since their origin. The first models were shallow and relied on the n-gram technique that simply counted occurrences of words or phrases. Later on, neural language models came about, such as recurrent neural networks (RNNs, LSTMs and GRUs) that enabled more complex models to rise.

One of the most significant advancements in LLMs came about in 2017, where the "Transformer" architechture was introduced in the paper titled "Attention Is All You Need". It enabled the training of deep neural nets to become highly parallelizable as compared to previous techniques. The attention mechanism that was introduced not only enabled the development of LLMs with billions of parameters but also found its way into multiple other areas. With the ongoing R&D in this area, it is likely that models will continue to become even more powerful and versatile in the future.

BERT, which is one such model based on the transformer architecture, is what we will be looking at for demonstration and finetuning in this blog.

---

### What is Finetuning?

Finetuning is the process of taking a pre-trained language model and further training it on a specific task, such as sentiment analysis, to improve its performance on that particular task. This is achieved by taking the pre-trained model and training it on a smaller dataset that is specific to the task. It allows the model to adapt its existing knowledge rather than having to learn everything from scratch.

In this blog, we will be looking at finetuning BERT (a language model developed by Google), that has been pre-trained on a large corpus of text allowing it to capture a wide range of language patterns and contextual relationships. The task that we will be finetuning it on is Masked Language Modelling.

---

### What is Masked Language Modelling?

MLM is a task which involves masking/removing some of the words in a sentence and making the model predict those words based on remaining contextual information. Think of it like adding blanks to a sentence to create fill-in-the-blanks style questions. In BERT, this is used as a pre-training task to allow it to learn a general understanding of the language. Masking involves randomly or algorithmically replacing words with a special `[MASK]` token.

An example of masking is as follows:

**Original Text**: There are so many discoveries and advancements in the field of AI and ML these days. What a time to be alive!

**After Masking**: There are so many [MASK] and advancements in the field of AI and [MASK] these days. What a time to be [MASK]!

**Probable Mask Prediction**: There are so many [innovations] and advancements in the field of AI and [technology] these days. What a time to be [alive]!

---

### Getting started with BERT

Access to models like BERT has been made very easy over the years, thanks in large part to platforms like Hugging Face, which provides a user-friendly interface for accessing and utilizing pre-trained language models. They also offer a wide range of pre-trained models and tools for finetuning them on specific tasks. With Hugging Face, even those without a deep understanding of machine learning can easily use pre-trained models like BERT to improve the performance of their NLP applications.

Let's dive into some code and get our hands dirty with BERT. The prerequisites for this is that you need a working installation of Python and have some experience using it.

#### Step 1 - Installing Required Libraries and Basics

Start by installing all the required libraries that we will be using throught the example.

```
pip install torch 'transformers[torch]' datasets ipywidgets nltk matplotlib==3.6.0 seaborn uptrain
```

Create a Python file or Jupyter notebook with the following code:

```python
import json
from transformers import pipeline

masked_sentence = "There are so many [MASK] and advancements in the field of AI and [MASK] these days. What a time to be [MASK]!"
fill_masker = pipeline(model="distilbert-base-uncased")
print(json.dumps(fill_masker(masked_sentence), indent=2))
```

Look at how simple the code is for getting started with MLM!

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

---

### Finetuning Task

Since we'd like to finetune BERT, we need to define the task at which we want to make it better. Our goal is to bias the masked word predictions to have more positive sentiment in the context of product reviews at Nike. We can do so by providing the model some labeled data to retrain on, where each sentence has a corresponding positive or negative sentiment.

In simpler words, let's say we have the sentence - "Nike shoes are very [MASK]". The predictions that we want for the masked token here are "popular", "durable", "comfortable" as opposed to "expensive", "ugly", "heavy".

#### Step 2 - Collecting/Synthesizing Datasets

For this task, we need a dataset that contains sentence examples of the above format. We could use an online dataset but for the sake of this example, let's synthesize our own. To put it simply, we create sentence templates with predetermined structures and insert specific words like adjectives or products into them to generate sentences. While some of the resulting sentences may not make complete sense, it is not a significant issue for this task as long as we are conveying enough information to the model. We can perform the data synthesis using the code below.

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

To perform our data synthesis, we need some basic setup and helper functions. The code for helper functions isn't added here for the sake of clarity of this blog. However, you can find the entire code [here](https://github.com/uptrain-ai/uptrain/blob/main/examples/finetuning_LLM/). With that set up, we can synthesize our data.

<details>
<summary>Code</summary>
<br />

```python
SYNTHESIZED_DATASET_SIZE = 25000
uptrain_save_fold_name = "uptrain_smart_data_bert"
synthesized_data_csv = 'data.csv'
synthesized_data_json = 'data.json'

# Create our own dataset of reviews for different companies, products, etc.
dataset = create_sample_dataset(SYNTHESIZED_DATASET_SIZE)
df = pd.DataFrame(dataset['data'])
df.reset_index(drop=True, inplace=True)

df.to_csv(synthesized_data_csv)
create_dataset_from_csv(synthesized_data_csv, 'text', synthesized_data_json)

with open(synthesized_data_json) as file:
    dataset = json.loads(file.read())
```
</details>

Let's also create a list of sentences that we will use to test and evaluate our model's performance on before and after finetuning, similar to the initial example above.

<details>
<summary>Sentence List</summary>
<br />

```python
testing_texts = [
    "Nike shoes are very [MASK]",
    "Nike atheletic wear is known for being very [MASK]",
    "Nike [MASK] shoes are very comfortable",
    "Trousers and Hoodies made by [MASK] are not very expensive",
    "Nike tshirts are famous for being [MASK]"
]
```
</details>

#### Step 3 - Integrating UpTrain

Now that we have our dataset and list of sentences for testing, we could try finetuning our model but it will not really improve the performance much. The model will be fed both positive and negative sentiment data and it may not learn to prioritize the positive sentiment predictions as expected in our finetuning task. This is where UpTrain comes in - with just a few lines of code, one can define a "Signal" which can be used as a filter for the dataset. It has other use cases as well, which you can find by checking out the UpTrain repository. Let's take a look at how to use UpTrain signals.

In the code below, we define three functions. These functions are callbacks that UpTrain signals will use to determine whether some data from our dataset is relevant to our task at hand. We can chain multiple signals as well as mix and match them by using `&` (combining signals logically with the AND operator) and `|` (combining signals logically with the OR operator) operators.

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

#### Step 4 - Generating Finetuning Dataset using UpTrain

With the framework defined, we can process our dataset with UpTrain using the code below. It is a simple loop that iterates over all our data in the dataset and passes it into the framework.

<details>
<summary>Code</summary>
<br />

```python
for index, sample in enumerate(dataset['data']):
    if index % 500 == 0:
        print(f'Processed {index} samples')
    inputs = {'text': [sample['text']]}
    framework.log(inputs = inputs, outputs = None)

retraining_csv = uptrain_save_fold_name + '/1/smart_data.csv'
retraining_json = 'retrain_dataset.json'
create_dataset_from_csv(retraining_csv, 'text', retraining_json)
```

</details>

There are a few things to look at here. UpTrain provides different Monitors for monitoring performance, checking for data distribution shifts, and collecting edge cases to retrain upon, among other things. Here, we use the EDGE_CASE monitor and provide it with our signals. We also add a data integrity check to make sure that none of our data contains null values. All monitoring related activities will show up on UpTrain's live dashboard. Once processing of this part completes, we will have created a retraining dataset that contains only those sentences that satisfy the requirements of the signals above. This dataset is not only smaller that the original (whether it be synthesized or obtained from a real source) but also contains only specific data that is relevant to the finetuning task.

#### Step 5 - Finetuning the model

Now that we have our retraining dataset tailored to our objective, we can begin with the retraining/finetuning process. HuggingFace offers user-friendly APIs that simplify the training and finetuning of models. If you want to understand how this is done for the above example, please refer to the complete source code available at here [here](https://github.com/uptrain-ai/uptrain/blob/main/examples/finetuning_LLM/).

#### Step 6 - Evaluation and Visualization

After finetuning is performed, we can visualize the different metrics that are used to study the performance of large language models. Two such metrics that we'll take a look at are [training/validation loss](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets) and [perplexity](https://www.wikiwand.com/en/Perplexity).

**Loss Plot**

![loss.png](./loss.png)

**Perplexity Plot**

![perplexity.png](./perplexity.png)

The code for plotting can be found in the linked notebook above. We can see that the performance of the model on our finetuning dataset improves with time as the values for both the loss and perplexity decrease. This is great!

Let's see what the predictions of the model on our pre-defined test sentences and verify whether the model has become better at our defined finetuning task or not.

<details>
<summary>Predictions</summary>
<br />

```
                Text: Nike shoes are very [MASK]
 Original Top tokens: ['popular', 'durable', 'expensive', 'comfortable', 'fashionable']
Finetuned Top tokens: ['durable', 'versatile', 'fashionable', 'efficient', 'comfortable']

                Text: Nike atheletic wear is known for being very [MASK]
 Original Top tokens: ['durable', 'expensive', 'popular', 'fashionable', 'rare']
Finetuned Top tokens: ['durable', 'fashionable', 'lightweight', 'efficient', 'flexible']

                Text: Nike [MASK] shoes are very comfortable
 Original Top tokens: ['polo', 'golf', 'swim', 'tennis', 'nike']
Finetuned Top tokens: ['training', 'basketball', 'soccer', 'running', 'football']

                Text: Trousers and Hoodies made by [MASK] are not very expensive
 Original Top tokens: ['women', 'manufacturers', 'men', 'amateurs', 'slaves']
Finetuned Top tokens: ['nike', 'honda', 'samsung', 'yamaha', 'bmw']

                Text: Nike tshirts are famous for being [MASK]
 Original Top tokens: ['.', ':', ';', 'colorful', 'unique']
Finetuned Top tokens: ['inexpensive', 'cheap', 'lightweight', 'versatile', 'durable']
```

</details>

We can see from the above output that the model is doing much better at predicting masked words that have positive sentiment associated with them. In the first example, we see that the model no longer predicts masked words such as "expensive" among its top 5 predictions, which is exactly what we want! An interesting case to look at is the predictions for example 3. We no longer have swim shoes!

We can also visualize the mask prediction confidence for different predictions, which is a helpful way of visualizing what words the model thinks are good substitutions. The code for this can be found in the linked notebook.

![predictions.png](./predictions.png)

---

And with that, we have learnt how to finetune a Large Language Model as well as how to use UpTrain for improving model performance and monitoring!

In conclusion, finetuning large language models like BERT can be a powerful tool for solving specific natural language processing tasks. UpTrain provides a simple and easy to use interface that requires minimal code to perform model monitoring, data drift checks, data integrity checks, edge case detection, model bias tracking, custom metric monitoring and much more. Checkout the UpTrain repository [here](https://github.com/uptrain-ai/uptrain).
