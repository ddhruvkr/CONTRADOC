import json
import random
import re
import ast
import torch
from evaluate import load
import numpy as np
bertscore = load("bertscore",device=torch.device("cpu"),num_process=2)

def load_docs_evis_infos(path, pos_only=False):
    with open(path) as f:
        data = json.load(f)
        positive_data = data["pos"]
        print("positive examples in total", len(positive_data))
        docs, evidences, info = [], [], []
        for key, value in positive_data.items():
            positive_data[key]["label"] = True
            positive_data[key]["id"] = key
            info.append(positive_data[key])
            docs.append(positive_data[key]["text"])
            evidences.append(positive_data[key]["evidence"])
        if pos_only:
            assert len(docs) == len(evidences) == len(info)
            return docs, evidences, info

        negative_data = data["neg"]
        print("negative examples in total", len(negative_data))
        for key, value in negative_data.items():
            negative_data[key]["label"] = False
            negative_data[key]["evidence"] = None
            negative_data[key]["id"] = key
            info.append(negative_data[key])
            docs.append(negative_data[key]["text"])
            evidences.append(negative_data[key]["evidence"])
    assert len(docs) == len(evidences) == len(info)
    return docs, evidences, info

def load_prompted_doc(list_of_doc, bs_prompt="", es_prompt=""):
    prompts = []
    for ele in list_of_doc:
        if ele[-1] != "\n": ele += "\n"
        prompts.append(bs_prompt+ele+es_prompt)
    return prompts
    
# The followings are metrics
def trim_string(s):
    start = 0
    end = len(s) - 1
    while start <= end and (s[start] == " " or s[start] == "\n"):
        start += 1
    while end >= start and (s[end] == " " or s[end] == "\n"):
        end -= 1
    return s[start:end + 1]

def direct_find(sents, query):
    if len(query) < 20: return None
    for i in range(len(sents)):
        if len(sents[i]) < 20 : continue
        if sents[i] in query or query in sents[i]: return i
    return None

def convert_string_to_list(strin:str):
    strin = trim_string(strin)
    strin = strin.replace("\n", "")
    match = re.search(r'\[.*\]', strin)
    if match:
        # Convert the string representation of a list to an actual list object
        try:
            evidence_list = ast.literal_eval(match.group(0))
            if isinstance(evidence_list, list):
                return evidence_list
            else:
                raise ValueError("The input is not a valid list.")
        except (ValueError, SyntaxError) as e:
            print("Error:", e)
            print(strin)
            return None

def extract_info(full_response:str):
    info = {"response": full_response}
    full_response = full_response.lower()
    beginning = full_response.find("\njudgment") + len("\njudgment")
    ending = full_response.find("\n", beginning)
    judge = full_response[beginning: ending]
    if "yes" in judge:
        info["judgment"] = "yes"
        try:
            evidence = convert_string_to_list(full_response.split("evidence:")[1])
        except:
            evidence = []
        if evidence is None: evidence = []
        info["evidence"] = evidence
        if evidence == []:
            info = extract_info_unstructured(full_response)
            print("Error! Answer Yes but no evidence correctly processed.")
    else:
        try:
            assert "no" in judge
        except:
            print(judge, full_response)
        info["judgment"] = "no"
        info["evidence"] = []
    return info

def extract_info_yes_no(full_response:str):
    info = {"response": full_response, "evidence": []}
    full_response = full_response.lower()
    info["judgment"]="yes"
    if "yes" in full_response:
        info["judgment"]="yes"
    else: info["judgment"]="no"
    return info

def extract_info_unstructured(full_response:str, with_ev=False):
    # TODO: copied sentences are in lines
    def remove_quote(s):
        if s[:2] == "- ":s = s[2:]
        try:
            #processed_item = re.sub(r'^\d+([.)]+)?\s*', '', item)
            s = re.sub(r'^\d+\.\s*', '', s)
            if s[0] in ["\"", "\'", " "]: s = s[1:]
            if s[-1] in ["\"", "\'", " "]: s = s[:-1]
        except: print("What happened?this is short", s)
        return s
    info = {"response": full_response}
    full_response = full_response.replace("\n\n","\n").replace("\n\n","\n").split("\n")
    info["evidence"] = [remove_quote(s) for s in full_response]
    info["judgment"] = "yes"
    return info

def calculate_metrics(gold_labels, predicted_labels, full_info):
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("The length of gold_labels and predicted_labels must be the same.")

    TP = FP = TN = FN = 0
    statistics_set = {"TP":[], "TN":[], "FN":[], "FP":[]}
    for gold, predicted, info_ele in zip(gold_labels, predicted_labels, full_info):
        if gold and predicted:
            TP += 1
            statistics_set["TP"].append(info_ele)
        elif not gold and not predicted:
            TN += 1
            statistics_set["TN"].append(info_ele)
        elif gold and not predicted:
            FN += 1
            statistics_set["FN"].append(info_ele)
        elif not gold and predicted:
            FP += 1
            statistics_set["FP"].append(info_ele)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy,
        'True Positive': TP,
        'False Positive': FP,
        'True Negative': TN,
        'False Negative': FN,
        # 'Statistic': statistics_set
    }
def batch_bs(list_of_found_evs, gold_ev, batch_size=32):
    num_batches = len(list_of_found_evs) // batch_size + (1 if len(list_of_found_evs) % batch_size else 0)
    bertscores_all_pr = []
    bertscores_all_recall = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size if i != num_batches - 1 else len(list_of_found_evs)

        batch_sents = list_of_found_evs[start:end]
        batch_statements = [gold_ev for _ in range(len(batch_sents))]

        results = bertscore.compute(predictions=batch_statements, references=batch_sents, lang="en", device=device)
        bertscores_pr = results["precision"]
        bertscore_recall = results["recall"]
        bertscores_all_pr.append(bertscores_pr)
        bertscores_all_recall.append(bertscore_recall)
        # entailment_scores_all.append(entailment_scores)
        # Freeing up memory
        torch._C._mps_emptyCache()
    bertscores_all_pr = np.concatenate(bertscores_all_pr)
    bertscores_all_recall = np.concatenate(bertscores_all_recall)
    torch._C._mps_emptyCache()
    return np.max(bertscores_all_pr).item(), np.max(bertscores_all_recall).item()


def yes_verify(list_of_found_evs, gold_ev, only_top3=False):
    if len(list_of_found_evs) == 0: return False
    list_of_found_evs = [x.lower() for x in list_of_found_evs]
    if only_top3 and len(list_of_found_evs)>3: list_of_found_evs = list_of_found_evs[:3]
    gold_ev = gold_ev.lower()
    if direct_find(list_of_found_evs, gold_ev) is not None: return True
    else:
        max_pr, max_rc = batch_bs(list_of_found_evs, gold_ev)
        if max_pr > 0.98 or max_rc > 0.98: return True
    return False

prompt_level1={"beginning":"""The task is to determine whether the article contains any self-contradictions. If yes, provide evidence by quoting mutually contradictory sentences in a list of strings in Python. If no, give an empty list.

Article:  """,
"end": """
Response: Form your answer in the following format (OR options are provided):

Judgment: yes OR no
Evidence: ["sentence1", "sentence2", ..., "sentenceN"] OR []""",
"pos_only": False,
"extract": extract_info
}

prompt_level2 = {"beginning":"""Self-Contradictory Article: An article is deemed self-contradictory when it contains one(self-conflict mention) or more statements that conflict with each other, making them mutually exclusive. The following article contains one self-contradiction. The task is to find where it is. Provide evidence by quoting mutually contradictory sentences from the article.

Article: """,
"end":"""

Please respond by giving 5 most possible sentences can reflect article-level contradiction(s), ranked by possibility high to low, separated with "\n". Don't explain.""",
                 "pos_only": True,
                 "extract": extract_info_unstructured}

prompt_yes_no = {"beginning": "", "end":"Determine whether the given article contains any self-contradictions. Only answer \"yes\" or \"no\"!","pos_only": False, "extract": extract_info_yes_no}
prompt_list_topk = {"beginning":"", "end": "The given article above contains one self-contradiction in it. Please list sentences contradict each other. Don't include anything else other than the exact sentences from the article.","pos_only": True,"extract": extract_info_unstructured}



if __name__ == "__main__":
    chose_prompt = prompt_yes_no
    debug = True
    docs, evidences, info = load_docs_evis_infos("ContraDoc.json",
                                                 pos_only=chose_prompt["pos_only"])
    prompts = load_prompted_doc(docs, bs_prompt=chose_prompt["beginning"], es_prompt=chose_prompt["end"])
    # Gather whatever responses as a list of strings, corresponding the order of the prompts.

    # TODO: Implement Code to get responses from models. Should corresponds to prompts one by one.
    responses = ["<placeholder of model responses>" for i in range(len(prompts))]

    assert len(prompts) == len(evidences) == len(info) == len(responses)

    verified_predictions, non_verified_predictions = [], []

    for i in range(len(prompts)):
        response_info = chose_prompt["extract"](responses[i].strip())
        non_verified_predictions.append(True if response_info["judgment"]=="yes" else False)
        if non_verified_predictions[-1] and info[i]["label"]: # only verify when both true
            verify = yes_verify(response_info["evidence"], evidences[i])
            if verify: verified_predictions.append(True)
            else: verified_predictions.append(False)
        else: verified_predictions.append(non_verified_predictions[-1])
    without_verify_res = calculate_metrics([x["label"] for x in info],
                                       non_verified_predictions, info)
    verify_res = calculate_metrics([x["label"] for x in info], verified_predictions, info)

    print("without_verify: ", without_verify_res)
    print("verify: ", verify_res)