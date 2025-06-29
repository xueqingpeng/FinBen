# %%
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login
from dotenv import load_dotenv
import os


# %%
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("using HF_TOKEN from .env")
else:
    print("warning: HF_TOKEN not found in .env")


# %%
gr_dataset = load_dataset("TheFinAI/plutus-finner-numeric")
en_dataset = load_dataset("TheFinAI/plutus-finner-numeric-english")

gr_test = gr_dataset['test']
en_test = en_dataset['test']

print(f"original greek dataset size: {len(gr_test)}")
print(f"translated english dataset size: {len(en_test)}")


# %%
translation_data = []

for i in range(min(len(gr_test), len(en_test))):
    gr_text = gr_test[i]['text']
    en_text = en_test[i]['text']
    
    translation_data.append({
        'text': gr_text,
        'answer': en_text,
        'query': f"Translate the following Greek text into English. Provide only the translation, without any additional text or commentary. Text to translate: {gr_text}"
    })


# %%
translation_dataset = Dataset.from_list(translation_data)
print(f"created {len(translation_dataset)} translation pairs")

print("\nfirst 3 examples:")
for i in range(3):
    print(f"greek: {translation_dataset[i]['text']}")
    print(f"english: {translation_dataset[i]['answer']}")
    print("---")


# %%
dataset_dict = DatasetDict({
    'train': translation_dataset,
    'test': translation_dataset,
    'validation': translation_dataset
})

print(f"Dataset structure: {dataset_dict}")

dataset_dict.push_to_hub(
    "TheFinAI/plutus-translation",
    private=False,
    commit_message="Add Greek-English translation dataset with all data in test split"
)

print("Dataset uploaded successfully!")
print("Dataset available at: https://huggingface.co/datasets/TheFinAI/plutus-translation")


# %%
