from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv() 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

file = client.files.create(
  file=open("data.jsonl", "rb"),
  purpose="fine-tune"
)

print(file)

ft_job = client.fine_tuning.jobs.create(
  training_file="file-SgGq0yqGOyjsef1MmptXKCFx", 
  model="gpt-3.5-turbo"
)

print(ft_job) # ftjob-dTFtmlO2uiLJN6V6qgEIDJ8K

# Retrieve the state of a fine-tune
print(client.fine_tuning.jobs.retrieve("ftjob-dTFtmlO2uiLJN6V6qgEIDJ8K"))

# List up to 10 events from a fine-tuning job
print(client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-dTFtmlO2uiLJN6V6qgEIDJ8K", limit=10))