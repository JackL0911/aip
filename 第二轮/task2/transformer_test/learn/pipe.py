from transformers import pipeline


pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0)

print(pipe)
print(pipe("it is bad for me"))

print(1)