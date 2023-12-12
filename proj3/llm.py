import os, re, json, glob, openai, faulthandler

from retry import retry
from string import Template
from neo4j import GraphDatabase
from timeit import default_timer as timer
# from dotenv import load_dotenv
from time import sleep
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained('bigscience/bloom-7b1')
tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-7b1')

generator = pipeline('text-generation', model = model, tokenizer=tokenizer)
code_prompt = 'Write python code to create a barplot'
result = generator(code_prompt, max_length = 200)

print(result)