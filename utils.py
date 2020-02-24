import os
import torch
import re

import json
import copy

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def ReadFile(FileName):
    FileName = "../Proposal/preprocess/" + str(FileName)

    text=[]
    ans=[]
    contents=[]
    with open(FileName, 'r') as InputFile:
        content = InputFile.read().split("\n")
        for index, Line in enumerate(content):
            if Line == "Story":
                text = re.split('[.?]', content[index+1])

            elif Line == "Answer":
                ans = content[index+1].split(".")
                contents.append([text, ans])
        contents.append([text, ans])

    return contents

def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    features=[]
    for pair in examples :
        for sen in pair[0]:
            inputs = tokenizer.encode_plus(sen, pair[1], add_special_tokens=True, max_length=max_length,)
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
           
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            padding_length = max_length - len(input_ids)

            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            
            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

            features.append(
              InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
           ))
    return features

if __name__ == "__main__":
    print("hu")
