import argparse
from tools import convert_string_to_list, load_narrative, load_doc_news, load_wiki
import json
from gpt_caller import get_responses_from_gpt
from prompts import insert_start_phrase, subjective_start_phrase_replace, replace_start_phrase

def parse_args():
    parser = argparse.ArgumentParser(description='GPT fake document generator')
    # Optional arguments with defaults
    parser.add_argument('--model', type=str, default='gpt-4',
                        choices=['gpt-4', 'gpt-3.5-turbo'],
                        help='GPT model to use (default: gpt-4)')
    parser.add_argument('--temperature', type=float, default=0,
                        help='Sampling temperature (default: 0)')
    parser.add_argument('--source-doc', type=str, default='cnn_dailymail',
                        choices=['cnn_dailymail', 'wikipedia', 'narrative'],
                        help='which type of document sources to use')
    parser.add_argument('--num-of-docs', type=int, default=10,
                        help='How many documents from the source to use')
    parser.add_argument('--subset', type=str, default="validation",
                        choices=['train', 'validation','test'],
                        help='which subset to use')
    parser.add_argument('--extract-method', type=str, default='content_insert',
                        choices=['content_insert','content_replace','subjective_replace'],
                        help='which technique to use, content-insert: generate factual content to insert to the document;'
                             'content_replace: generate factual content to replace; subjective_replace:generate perspective/mood to replace')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.extract_method == "content_insert": start_phrase = insert_start_phrase
    elif args.extract_method == "content_replace": start_phrase =  replace_start_phrase
    elif args.extract_method == 'subjective_replace': start_phrase = subjective_start_phrase_replace

    if args.source_doc == "cnn_dailymail": loader = load_doc_news
    elif args.source_doc == "wikipedia": loader = load_wiki
    elif args.source_doc == "narrative": loader = load_narrative
    prompts, id_list = loader(start_phrase, args.num_of_docs, args.subset)
    print(prompts[0])
    preds = get_responses_from_gpt(prompts, model=args.model, temp=args.temperature)
    full_preds = [convert_string_to_list(x.replace("\n","")) for x in preds] # json dict
    assert len(full_preds) == len(id_list)
    for i in range(len(full_preds)):
        full_preds[i] = {"statements": full_preds[i], "info": {"doc_source":f"{args.source_doc} {args.subset}","doc_id": id_list[i]}}
    with open(f"contradiction_json/contradiction_{args.source_doc}_{args.subset}_{args.num_of_docs}.json", "w") as f:
        json.dump(full_preds, f)