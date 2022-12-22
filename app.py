# --------------------------------------- IMPORTS ----------------------------------------------

import os
import argparse
from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead


# ==============================================================================================
# ----------------------------- LOADING TRAINED MODEL -------------------------------------

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=5000)
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

# Load trained GPT2 model and tokenizer
model_dir = 'model/'
tokenizer_dir = 'tokenizer/'
device = torch.device(args.device)
model = AutoModelWithLMHead.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model.to(device)
model.eval()
# ===============================================================================================
# ------------------------------ PREPROCESS AND INFERENCE FUNCTIONS -----------------------------

def get_poem(prompt, model=model, tokenizer=tokenizer):
    prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    sample_outputs = model.generate(input_ids,
                                   do_sample=True, 
                                   max_length=512, 
                                   min_length=512,
                                   num_return_sequences=1)
    poem = tokenizer.decode(sample_outputs[0])

    return poem

def replace_nth(s, sub, repl, n=1):
    chunks = s.split(sub)
    size = len(chunks)
    rows = size // n + (0 if size % n == 0 else 1)
    return repl.join([
        sub.join([chunks[i * n + j] for j in range(n if (i + 1) * n < size else size - i * n)])
        for i in range(rows)
    ])

def postprocess(poem_string):
    poem_string = poem_string.split('.')[0]
    poem_string = poem_string.split('  ')
    num_verses = min(19, len(poem_string))
    if (num_verses % 2) != 0:
        num_verses -= 1
    poem = ''
    for i in range(num_verses):
        poem = poem + poem_string[i] + ' | '
    poem = poem.replace('  ',' | ')
    poem = replace_nth(poem , '|', '<br />', n=2).replace('|','    ')
    return poem

# ===============================================================================================
# ----------------------------------------- FLASK APP -------------------------------------------


app = Flask(__name__)

# render home page

@ app.route('/')
def home():
    title = 'Arabic Poem Generator'
    return render_template('index.html', title=title)


# ===============================================================================================
# ------------------------------------ RENDER PREDICTION PAGE -----------------------------------

@ app.route('/', methods=['POST'])
def generate():
    title = 'Arabic Poem Generator'

    prompt = request.form.get('prompt')
    poem = get_poem(prompt)
    poem = postprocess(poem)

    return render_template('result.html', poem=poem, title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port)