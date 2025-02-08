from datasets import load_dataset
import tiktoken
import ast

def convert_string_to_list(string):
    try:
        # Safely evaluate the string and convert it back to a list
        result = ast.literal_eval(string)
        if isinstance(result, list):
            return result
        else:
            raise ValueError("The input is not a valid list.")
    except (SyntaxError, ValueError) as e:
        print("Error:", e)
        return None


def load_doc_news(start_phrase, count_num=10, subset="validation"):
    enc = tiktoken.get_encoding("cl100k_base")
    assert enc.decode(enc.encode("hello world")) == "hello world"
    dataset = load_dataset("cnn_dailymail", '3.0.0')
    prompts, id_list = [], []
    count, id_n = 0, 0
    for ele in  dataset[subset]:
        id_n += 1
        if count >= count_num: break
        a = len(enc.encode(ele["article"]))
        if a < 400: continue
        prompts.append(start_phrase + ele["article"])
        id_list.append(id_n)
        count += 1
    return prompts, id_list
def load_arxiv_pubmed(start_phrase, count_num=10, choice="arxiv", subset="validation"):
    enc = tiktoken.get_encoding("cl100k_base")
    assert enc.decode(enc.encode("hello world")) == "hello world"
    prompts, id_list = [], []
    count, id_n = 0, 0
    dataset = load_dataset("scientific_papers", choice)
    for example in dataset[subset]:
        id_n += 1
        if count >= count_num: break
        a = len(enc.encode(example["article"]))  # 304 docs are averaged as 1391/6400 tokens arxiv
        # 1770/6658 docs are 1364 tokens pubmed
        if 500 < a < 1500:
            prompts.append(start_phrase + example["article"])
            count += 1
            id_list.append(id_n)
    return prompts, id_list

def load_wiki(start_phrase, count_num=10, subset="validation"):
    enc = tiktoken.get_encoding("cl100k_base")
    assert enc.decode(enc.encode("hello world")) == "hello world"
    prompts, id_list = [], []
    count, id_n = 0, 0
    dataset = load_dataset("EleutherAI/wikitext_document_level", "wikitext-103-raw-v1")

    for example in dataset[subset]:
        if count >= count_num: break
        a = len(enc.encode(example["page"]))
        if 500 < a < 1600:
            prompts.append(start_phrase + example["page"].replace("@-@", "-").replace("@.@", ".").replace(" .", "."))
            count += 1
            id_list.append(id_n)
        id_n += 1
    return prompts, id_list

def load_narrative(start_phrase, count_num=10, subset="validation"):
    enc = tiktoken.get_encoding("cl100k_base")
    assert enc.decode(enc.encode("hello world")) == "hello world"
    prompts, id_list = [], []
    count, id_n = 0, 0
    dataset = load_dataset("narrativeqa")
    doc_id_list = set()
    for example in dataset[subset]:
        id_n += 1
        if count >= count_num: break
        if example["document"]["kind"] != "gutenberg" or example["document"]["id"] in doc_id_list: continue
        summary = example["document"]["summary"]["text"]
        if summary.lower().count("the play") >= 2 or len(example["document"]["summary"]["tokens"]) < 300: continue
        doc_id_list.add(example["document"]["id"])
        prompts.append(start_phrase + summary)
        count += 1
        id_list.append(id_n)
    return prompts, id_list

def prompt_wrap(prompts, system_msg=None, has_init=True):
    full_prompts = []
    init = {"role": "system", "content": system_msg if system_msg is not None else "You are a helpful assistant."}
    for prompt in prompts:
        messages = [init] if has_init else []
        messages.append({"role": "user", "content": prompt})
        full_prompts.append(messages)
    return full_prompts # return list of messages

