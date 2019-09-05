"""
Script to convert the retrieved HITS into an entailment dataset
USAGE:
 python scripts/convert_to_entailment.py hits_file output_file

JSONL format of files
 1. hits_file:
 {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "George wants to warm his hands quickly by rubbing them. Which skin surface will
               produce the most heat?"
      "choice": {"text": "dry palms", "label": "A"},
       "support": {
         "text": "Use hand sanitizers according to directions, which usually involves rubbing for
                  at least ten seconds, then allowing hands to air dry."
         ...
        }
    },
     "answerKey":"A"
  }

 2. output_file:
   {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "George wants to warm his hands quickly by rubbing them. Which skin surface will
               produce the most heat?"
      "choice": {"text": "dry palms", "label": "A"},
      "support": {
         "text": "Use hand sanitizers according to directions, which usually involves rubbing for
                  at least ten seconds, then allowing hands to air dry."
         ...
        }
    },
     "answerKey":"A",
     "premise": "Use hand sanitizers according to directions, which usually involves rubbing for
                  at least ten seconds, then allowing hands to air dry.",
     "hypothesis": "George wants to warm his hands quickly by rubbing them. Dry palms skin
                    surface will produce the most heat."
  }
"""

import json
import re
import sys

from allennlp.common.util import JsonDict

# String used to indicate a blank
BLANK_STR = "___"


def convert_to_entailment(qa_file: str, output_file: str):
    with open(output_file, 'w') as output_handle, open(qa_file, 'r') as qa_handle:
        print("Writing to {} from {}".format(output_file, qa_file))
        for line in qa_handle:
            json_line = json.loads(line)
            output_dict = convert_qajson_to_entailment(json_line)
            output_handle.write(json.dumps(output_dict))
            output_handle.write("\n")


# Convert the QA file json to output dictionary containing premise and hypothesis
def convert_qajson_to_entailment(qa_json: JsonDict):
    question_text = qa_json["question"]["stem"]
    choice = qa_json["question"]["choice"]["text"]
    support = qa_json["question"]["support"]["text"]
    hypothesis = create_hypothesis(get_fitb_from_question(question_text), choice)
    output_dict = create_output_dict(qa_json, support, hypothesis)
    return output_dict


# Get a Fill-In-The-Blank (FITB) statement from the question text. E.g. "George wants to warm his
# hands quickly by rubbing them. Which skin surface will produce the most heat?" ->
# "George wants to warm his hands quickly by rubbing them. ___ skin surface will produce the most
# heat?
def get_fitb_from_question(question_text: str) -> str:
    fitb = replace_wh_word_with_blank(question_text)
    if not re.match(".*_+.*", fitb):
        print("Can't create hypothesis from: '{}'. Appending {} !".format(question_text, BLANK_STR))
        # Strip space, period and question mark at the end of the question and add a blank
        fitb = re.sub("[\.\? ]*$", "", question_text.strip()) + BLANK_STR
    return fitb


# Create a hypothesis statement from the the input fill-in-the-blank statement and answer choice.
def create_hypothesis(fitb: str, choice: str) -> str:
    if ". " + BLANK_STR in fitb or fitb.startswith(BLANK_STR):
        choice = choice[0].upper() + choice[1:]
    else:
        choice = choice.lower()
    # Remove period from the answer choice, if the question doesn't end with the blank
    if not fitb.endswith(BLANK_STR):
        choice = choice.rstrip(".")
    # Some questions already have blanks indicated with 2+ underscores
    hypothesis = re.sub("__+", choice, fitb)
    return hypothesis


# Identify the wh-word in the question and replace with a blank
def replace_wh_word_with_blank(question_str: str):
    wh_word_offset_matches = []
    wh_words = ["which", "what", "where", "when", "how", "who", "why"]
    for wh in wh_words:
        # Some Turk-authored SciQ questions end with wh-word
        # E.g. The passing of traits from parents to offspring is done through what?
        m = re.search(wh + "\?[^\.]*[\. ]*$", question_str.lower())
        if m:
            wh_word_offset_matches = [(wh, m.start())]
            break
        else:
            # Otherwise, find the wh-word in the last sentence
            m = re.search(wh + "[ ,][^\.]*[\. ]*$", question_str.lower())
            if m:
                wh_word_offset_matches.append((wh, m.start()))

    # If a wh-word is found
    if len(wh_word_offset_matches):
        # Pick the first wh-word as the word to be replaced with BLANK
        # E.g. Which is most likely needed when describing the change in position of an object?
        wh_word_offset_matches.sort(key=lambda x: x[1])
        wh_word_found = wh_word_offset_matches[0][0]
        wh_word_start_offset = wh_word_offset_matches[0][1]
        # Replace the last question mark with period.
        question_str = re.sub("\?$", ".", question_str.strip())
        # Introduce the blank in place of the wh-word
        fitb_question = (question_str[:wh_word_start_offset] + BLANK_STR +
                         question_str[wh_word_start_offset + len(wh_word_found):])
        # Drop "of the following" as it doesn't make sense in the absence of a multiple-choice
        # question. E.g. "Which of the following force ..." -> "___ force ..."
        return fitb_question.replace(BLANK_STR + " of the following", BLANK_STR)
    elif re.match(".*[^\.\?] *$", question_str):
        # If no wh-word is found and the question ends without a period/question, introduce a
        # blank at the end. e.g. The gravitational force exerted by an object depends on its
        return question_str + " " + BLANK_STR
    else:
        # If all else fails, assume "this ?" indicates the blank. Used in Turk-authored questions
        # e.g. Virtually every task performed by living organisms requires this?
        return re.sub(" this[ \?]", " ___ ", question_str)


# Create the output json dictionary from the input json, premise and hypothesis statement
def create_output_dict(input_json: JsonDict, premise: str, hypothesis: str) -> JsonDict:
    input_json["premise"] = premise
    input_json["hypothesis"] = hypothesis
    return input_json


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Provide at least two arguments: "
                         "json file with hits, output file name")
    convert_to_entailment(sys.argv[1], sys.argv[2])
