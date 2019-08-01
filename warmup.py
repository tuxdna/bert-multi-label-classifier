from bert_serving.client import BertClient
import tqdm

bc = BertClient(check_length=False, timeout=2000)

print("Example of how BERT encoding works with input text:")
txt = "Welcome to the DevConf2019 conference! Bangalore has awesome weather today."
embedding, tokens = bc.encode([txt], show_tokens=True)
print("Type and shape of returned embedding", type(embedding), embedding.shape)
print(tokens)

encodings = []

print("Running it 100 times more to warmup the server...")
for i in tqdm.tqdm(range(0, 100)):
    x = "Hello World %s !" % i
    enc = bc.encode([x])
    encodings.append(enc)
