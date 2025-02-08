from datasets import load_dataset
import json
import spacy
from transformers import BertTokenizer, BertForNextSentencePrediction
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from evaluate import load
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch.nn.functional as F
import numpy as np
from sklearn import preprocessing
from collections import defaultdict
import random

content_json_dir = "contradiction_json/cnn_cont_test5.json" # both replacement/insertion
mood_json_dir = "contradiction_json/cnn_mood_test5.json" # for replacement only

def reduce_list_to_n(l, n):
    if n >= len(l):
        return l
    selected_indices = sorted(random.sample(range(len(l)), n))
    return [l[i] for i in selected_indices]

def token_overlap_rate(ref_sent, target_sent):
    ref_tokens = ref_sent.lower().split()
    target_tokens = target_sent.lower().split()

    not_found_count = 0

    for token in target_tokens:
        if token in ref_tokens:
            ref_tokens.remove(token)
        else:
            not_found_count += 1

    return not_found_count / len(target_tokens)

def cal_num(data):
    num = 0
    for ele in data:
        num += len(ele["statements"])
    return num
def load_doc_news(doc_id, split="validation"):
    dataset = load_dataset("cnn_dailymail", '3.0.0')
    this = dataset[split][doc_id-1]
    return this["article"]

def load_doc_wiki(doc_id, split="validation"):
    dataset = load_dataset("EleutherAI/wikitext_document_level", "wikitext-103-raw-v1")
    this = dataset[split][doc_id]
    return this["page"].replace("@-@", "-").replace("@.@", ".").replace(" .", ".")

def load_doc_story(doc_id, split="validation"):
    dataset = load_dataset("narrativeqa")
    this = dataset[split][doc_id-1]
    return this["document"]["summary"]["text"]

def load_doc_papers(doc_id, split="validation", subset="pubmed"):
    dataset = load_dataset("scientific_papers", subset)
    this = dataset[split][doc_id-1]
    return this["article"]

def load_data(doc_source, doc_id, subset="arxiv"):
    all_doc_source_domain = doc_source.split(", ")
    if all_doc_source_domain[0] == "cnn_dailymail":
        this_text = load_doc_news(doc_id, all_doc_source_domain[1])
    return this_text

def combine_statements(lists_of_dicts):
    combined = defaultdict(list)

    # Iterate through the lists of dictionaries
    for list_of_dicts in lists_of_dicts:
        for item in list_of_dicts:
            doc_id = item['info']['doc_id']
            # Append the statements to the corresponding doc_id
            combined[doc_id].extend(item['statements'])

    # Convert the result into the desired format
    result = [{"statements": statements, "info": {"doc_id": doc_id, "source": "source_doc"}}
              for doc_id, statements in combined.items()]

    return result


with open(content_json_dir) as f2:
    content_statement = json.load(f2)

with open(mood_json_dir) as f1:
    replacement = json.load(f1)
    replacement = combine_statements([replacement, content_statement])

print("number of statements original", cal_num(content_statement), cal_num(replacement))

loader = load_doc_news
ppl_threshold = 0.05 # This needs to modify according to the details in the paper
saver_all = {} # Doc_id

device = torch.device("cuda")
bertscore = load("bertscore",device=device,num_process=1)

nli_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base')
nli_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base')
nli_model.to(device)

nlp = spacy.load('en_core_web_sm')  # Load the English Model
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
model = BertForNextSentencePrediction.from_pretrained("bert-large-uncased")
model = model.to(device)

ppl_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
ppl_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)



def insert_position(sents, ele):
    logits = []
    for i in range(len(sents) - 1):
        former = " ".join([sents[i-1], sents[i]]) if i>=1 else sents[i]
        latter = " ".join([sents[i + 1], sents[i+2]]) if i < len(sents) - 2 else sents[i+1]
        encoding = tokenizer(former, ele, return_tensors="pt").to(device)
        outputs = model(**encoding, labels=torch.LongTensor([1]).to(device))
        if outputs.logits[0, 0] < outputs.logits[0, 1] or outputs.logits[0, 0] < 0 or outputs.logits[0, 1] > 0:
            logits.append(-1)
            continue
        logit1 = F.softmax(outputs.logits)[0, 0]
        encoding = tokenizer(ele, latter, return_tensors="pt").to(device)
        outputs = model(**encoding, labels=torch.LongTensor([1]).to(device))
        if outputs.logits[0, 0] < outputs.logits[0, 1] or outputs.logits[0, 0] < 0 or outputs.logits[
            0, 1] > 0: logits.append(-1); continue
        logit2 = F.softmax(outputs.logits)[0, 0]
        logits.append(logit1.item() + logit2.item())

    assert len(logits) == len(sents) - 1
    rank_places = list(np.argsort(np.array(logits)))
    rank_places.reverse()
    for name, var in locals().items():
        if torch.is_tensor(var):
            del var
    return rank_places


def insert_position_batchnize(sents, ele, batch_size=32):
    num_batches = (len(sents) - 1) // batch_size + (1 if (len(sents) - 1) % batch_size else 0)
    logits_all = []
    eles = [ele for _ in range(len(sents) - 1)]
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size if i != num_batches - 1 else len(sents) - 1
        batch_formers = []
        batch_latters = []
        for j in range(start, end):
            former = " ".join([sents[j - 1], sents[j]]) if j >= 1 else sents[j]
            latter = " ".join([sents[j + 1], sents[j + 2]]) if j < len(sents) - 2 else sents[j + 1]
            batch_formers.append(former)
            batch_latters.append(latter)

        batch_encoding = tokenizer(batch_formers, eles[start:end], return_tensors="pt", padding=True,
                                   truncation=True).to(device)
        batch_outputs = model(**batch_encoding, labels=torch.ones((end - start,), dtype=torch.int).to(device))
        logits = []
        for i in range(batch_outputs.logits.size(0)):
            if batch_outputs.logits[i, 0] < batch_outputs.logits[i, 1] or batch_outputs.logits[i, 0] < 0 or \
                    batch_outputs.logits[i, 1] > 0:
                logits.append(-1)
            else:
                logits.append(F.softmax(batch_outputs.logits, dim=-1)[i, 0].item())

        batch_encoding = tokenizer(eles[start:end], batch_latters, return_tensors="pt", padding=True,
                                   truncation=True).to(device)
        batch_outputs = model(**batch_encoding, labels=torch.ones((end - start,), dtype=torch.int).to(device))
        for i in range(batch_outputs.logits.size(0)):
            if logits[i] == -1: continue
            if batch_outputs.logits[i, 0] < batch_outputs.logits[i, 1] or batch_outputs.logits[i, 0] < 0 or \
                    batch_outputs.logits[i, 1] > 0:
                logits[i] = -1
            else:
                logits[i] += F.softmax(batch_outputs.logits, dim=-1)[i, 0].item()
        logits_all.extend(logits)
        # Freeing up memory
    assert len(logits_all) == len(sents) - 1
    rank_places = list(np.argsort(np.array(logits_all)))
    rank_places.reverse()

    return rank_places


def insert_position_batch(sents, ele):
    formers, latters, logits = [], [], []
    eles = [ele for _ in range(len(sents)-1)]
    for i in range(len(sents) - 1):
        former = " ".join([sents[i-1], sents[i]]) if i>=1 else sents[i]
        latter = " ".join([sents[i + 1], sents[i+2]]) if i < len(sents) - 2 else sents[i+1]
        formers.append(former)
        latters.append(latter)
    encoding = tokenizer(formers, eles, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**encoding, labels=torch.ones((len(sents)-1,),dtype=torch.int).to(device))
    for i in range(outputs.logits.size(0)):
        if outputs.logits[i, 0] < outputs.logits[i, 1] or outputs.logits[i, 0] < 0 or outputs.logits[i, 1] > 0:
            logits.append(-1)
        else: logits.append(F.softmax(outputs.logits)[i,0].item())

    encoding = tokenizer(eles, latters, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**encoding, labels=torch.ones((len(sents)-1,),dtype=torch.int).to(device))
    for i in range(outputs.logits.size(0)):
        if logits[i] == -1: continue
        if outputs.logits[i, 0] < outputs.logits[i, 1] or outputs.logits[i, 0] < 0 or outputs.logits[i, 1] > 0:
            logits[i]= -1
        else: logits[i] += F.softmax(outputs.logits)[i,0].item()

    assert len(logits) == len(sents) - 1
    rank_places = list(np.argsort(np.array(logits)))
    rank_places.reverse()
    for name, var in locals().items():
        if torch.is_tensor(var):
            del var
    return rank_places

def weighted(nsp=None, ppl=None, nli=None, berts=None):
    if nsp is not None:
        normalized_nsp = preprocessing.normalize([nsp])
    if ppl is not None:
        normalized_ppl = preprocessing.normalize([nsp])
    return normalized_nsp + normalized_ppl

def filter_non_counter(statements_dict):
    statements, contradictions = [], []
    for ele in statements_dict:
        statements.append(ele["Select Statement"])
        contradictions.append(ele["Contradicted Statement"])
    features = nli_tokenizer(statements, contradictions,padding=True, truncation=True, return_tensors="pt").to(device)
    entailment_scores = nli_model(**features).logits # n * 3 index 1 is entailment
    entailment_scores = F.softmax(entailment_scores)
    contradiction_scores = entailment_scores[:, 0]  # Get the scores for 'contradiction'
    contradiction_indices = (contradiction_scores > 0.9).nonzero(as_tuple=True)[0].detach().cpu().tolist()  # Get the indices where score > 0.5
    if len(contradiction_indices) < 10: print(contradiction_scores.detach().cpu().numpy())
    for name, var in locals().items():
        if torch.is_tensor(var):
            # print(f"Deleting {name}")
            del var
    return contradiction_indices

def find_ref_sent(sents, statement):
    statements= [statement for _ in range(len(sents))]
    results = bertscore.compute(predictions=statements, references=sents, lang="en", device=device)
    bertscores = results["precision"]
    features = nli_tokenizer(sents, statements, padding=True, truncation=True, return_tensors="pt").to(device)
    entailment_scores = nli_model(**features).logits # n * 3 index 1 is entailment
    entailment_scores = F.softmax(entailment_scores)[:,1].detach().cpu().numpy()
    for name, var in locals().items():
        if torch.is_tensor(var):
            # print(f"Deleting {name}")
            del var
    return np.argmax(bertscores), np.argmax(entailment_scores)

def find_ref_sent_batch(sents, statement, batch_size=32):
    num_batches = len(sents) // batch_size + (1 if len(sents) % batch_size else 0)
    bertscores_all = []
    entailment_scores_all = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size if i != num_batches - 1 else len(sents)

        batch_sents = sents[start:end]
        batch_statements = [statement for _ in range(len(batch_sents))]

        results = bertscore.compute(predictions=batch_statements, references=batch_sents, lang="en", device=device)
        bertscores = results["precision"]
        features = nli_tokenizer(batch_sents, batch_statements, padding=True, truncation=True, return_tensors="pt").to(device)
        entailment_scores = nli_model(**features).logits # n * 3 index 1 is entailment
        entailment_scores = F.softmax(entailment_scores, dim=-1)[:,1].detach().cpu().numpy()

        bertscores_all.append(bertscores)
        entailment_scores_all.append(entailment_scores)
        # Freeing up memory

    bertscores_all = np.concatenate(bertscores_all)
    entailment_scores_all = np.concatenate(entailment_scores_all)

    return np.argmax(bertscores_all).item(), np.argmax(entailment_scores_all).item()


def perplexity_checker(sent):
    # Load pre-trained model and tokenizer (GPT-2)
    input_ids = torch.tensor(ppl_tokenizer.encode(sent, add_special_tokens=True, max_length=1024,truncation=True)).unsqueeze(0).to(device)
    # Calculate loss
    with torch.no_grad():
        outputs = ppl_model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
    # Calculate perplexity
    perplexity = torch.exp(loss).item()
    for name, var in locals().items():
        if torch.is_tensor(var):
            del var
    return perplexity

def ppl_diff(original, insert):
    ppl1 = perplexity_checker(original)
    ppl2 = perplexity_checker(insert)
    return ppl2 - ppl1



ref_same = 0
doc_cnt = 0
avg_distance = 0
document_save = []
for statement_psg in content_statement:
    all_num = 0
    doc_id = statement_psg["info"]["doc_id"]
    original_passage = loader(doc_id, "test")
    doc = nlp(original_passage)
    sents = [sent.text for sent in doc.sents]
    original_passage = " ".join(sents)
    distance_rank = [0 for _ in range(5)]
    counter_indexes = filter_non_counter(statement_psg["statements"])
    valid_statement = []
    for _ in counter_indexes:
        valid_statement.append(statement_psg["statements"][_])
    print("after filtered:", len(valid_statement))
    for ele in valid_statement:
        # if ele.find(": ") != -1: ele = ele.split(": ")[1]
        bs_sent_id, nli_sent_id = find_ref_sent_batch(sents, ele["Original Sentence"])
        print("original_statement:", ele["Select Statement"])
        print("bertscore reference sentence:", sents[bs_sent_id])
        print("nli reference sentence:", sents[nli_sent_id])
        if bs_sent_id == nli_sent_id: ref_same += 1
        else:
            print("Didn't find the correct ref sentence.")
            print("--------------")
            continue
        pos_rank = insert_position_batchnize(sents, ele["Contradicted Statement"])
        best_place = [nli_sent_id, nli_sent_id]
        count = 0
        # Add second_best_place
        for pos in pos_rank:
            if pos != nli_sent_id and pos+1 != nli_sent_id:
                best_place[count] = pos
                count += 1
            if count> 1: break
        print("prev: " + sents[best_place[0]], "\ncounter: ", ele["Contradicted Statement"], "\nnext:", sents[best_place[0]+1])
        new_doc = sents.copy()
        new_doc.insert(best_place[0] + 1, ele["Contradicted Statement"])
        new_doc = " ".join(new_doc)

        ppl_difference = ppl_diff(original_passage, new_doc)

        if ppl_difference < 0.03:
            document_save.append(new_doc)

            header = "found sentence: " + sents[bs_sent_id] + "\n" \
                     + "statement: " + ele["Select Statement"] + "\n" \
                     + "position (char index): " + str(new_doc.find(ele["Contradicted Statement"])) + "\n" \
                     + "contradict statment:" + ele["Contradicted Statement"] + "\n------------------\n"
            print(header)
            all_num += 1
            if doc_id not in saver_all: saver_all[doc_id] = []
            saver_all[doc_id].append({
                "original_statement": ele["Select Statement"],
                "original_sentence": sents[bs_sent_id],
                "contradiction": ele["Contradicted Statement"],
                "full_text": new_doc,
                "original_position": bs_sent_id
            })
        else: print("abandon because ppl is too high")
        print("distance", best_place[0] - nli_sent_id, best_place[1] - nli_sent_id, "preplexity difference", ppl_difference)#, ppl_difference_2)
        #avg_distance += abs(best_place[0] - nli_sent_id)
        print("--------------")
print("insert num",sum([len(_) for _ in saver_all.values()]))

ref_same = 0
doc_cnt = 0
avg_distance = 0
document_save = []
for statement_psg in replacement:
    all_num = 0
    doc_id = statement_psg["info"]["doc_id"]
    original_passage = loader(doc_id, "test")
    doc = nlp(original_passage)
    sents = [sent.text for sent in doc.sents]
    original_passage = " ".join(sents)
    distance_rank = [0 for _ in range(5)]
    counter_indexes = filter_non_counter(statement_psg["statements"])
    valid_statement = []
    for _ in counter_indexes:
        valid_statement.append(statement_psg["statements"][_])
    print("after filtered:", len(valid_statement))
    for ele in valid_statement:
        # if ele.find(": ") != -1: ele = ele.split(": ")[1]
        bs_sent_id, nli_sent_id = find_ref_sent_batch(sents, ele["Original Sentence"])

        print("original_statement:", ele["Original Sentence"])
        print("bertscore reference sentence:", sents[bs_sent_id])
        print("Contradicted Sentence", ele["Rewritten Sentence"])

        if bs_sent_id == nli_sent_id: ref_same += 1
        else: continue
        new_doc = sents.copy()
        new_doc[bs_sent_id] = ele["Rewritten Sentence"]
        new_doc = " ".join(new_doc)
        # print(new_doc, original_passage)
        # break
        ppl_difference = ppl_diff(original_passage, new_doc)
        print("ppl difference", ppl_difference)
        if ppl_difference < 0.08:
            document_save.append(new_doc)
            # found sentence; statement; contradiction; insert place
            header = "found sentence: " + sents[bs_sent_id] + "\n" \
                     + "statement: " + ele["Select Statement"] + "\n" \
                     + "position (char index): " + str(new_doc.find(ele["Rewritten Sentence"])) + "\n" \
                     + "contradict statment:" + ele["Contradicted Statement"] + "\n------------------\n"
            print(header)
            if doc_id not in saver_all: saver_all[doc_id] = []
            saver_all[doc_id].append({
                "original_statement":  ele["Select Statement"],
                "original_sentence": sents[bs_sent_id],
                "contradiction": ele["Contradicted Statement"],
                "doc_before": " ".join(sents[:bs_sent_id]),
                "sent_rep": ele["Rewritten Sentence"],
                "doc_after": " ".join(sents[bs_sent_id+1: ]),
                "original_position": bs_sent_id
            })
            all_num += 1
        else: print("abandon because ppl is too high")
        print("--------------")
# Group by sentence reference
num = 0
for key in saver_all.keys():
    saver_all[key] = sorted(saver_all[key], key=lambda x: x['original_position'])
    saver_all[key] = reduce_list_to_n(saver_all[key], 6)
    num += len(saver_all[key])
print("accepted number", num)

with open("file_save.json","w") as f:
    json.dump(saver_all,f, indent=2)