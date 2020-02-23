import os
import torch
import re

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


