#============================================================

#COMPLAINT NEGATIVITY SCORER

#Pretrained: cardiffnlp/twitter-roberta-base-sentiment-latest

#Output: Negative sentiment score from 0.0 to 1.0

#============================================================

#---------- INSTALL DEPENDENCIES ----------

import sys
import subprocess

packages = ["torch", "transformers"]

for package in packages:
subprocess.check_call([
sys.executable,
"-m",
"pip",
"install",
package
])

#---------- IMPORTS ----------

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#---------- LOAD PRETRAINED MODEL ----------

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

print("\nLoading pretrained sentiment model...")
print("First run may take a few minutes while the model downloads.\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
MODEL_NAME
)

model.eval()

print("Model loaded successfully!")

#---------- FIND NEGATIVE LABEL ----------

negative_index = None

for index, label in model.config.id2label.items():
if label.lower() == "negative":
negative_index = int(index)
break

if negative_index is None:
raise RuntimeError("Could not find the negative label.")

#---------- NEGATIVITY FUNCTION ----------

def measure_negativity(complaint):

if not isinstance(complaint, str) or not complaint.strip():  
    raise ValueError("Please enter a valid complaint.")  

inputs = tokenizer(  
    complaint,  
    return_tensors="pt",  
    truncation=True,  
    max_length=128  
)  

with torch.no_grad():  
    outputs = model(**inputs)  

probabilities = torch.softmax(  
    outputs.logits,  
    dim=-1  
)[0]  

negative_score = probabilities[negative_index].item()  

# Guarantee the expected 0-1 range  
negative_score = max(0.0, min(1.0, negative_score))  

return negative_score

#============================================================

#ENTER A COMPLAINT

#============================================================

print("\n==========================================")
print("       COMPLAINT NEGATIVITY SCORER")
print("==========================================")
print("Enter a complaint and press Enter.")
print("Type 'exit' to stop.\n")

while True:

complaint = input("Complaint: ")  

if complaint.lower().strip() == "exit":  
    print("\nDone.")  
    break  

try:  

    score = measure_negativity(complaint)  

    print(  
        f"Negativity Score: {score:.4f}"  
    )  

    print(  
        f"Range: 0.0000 - 1.0000\n"  
    )  

except Exception as e:  

    print(f"Error: {e}\n")
