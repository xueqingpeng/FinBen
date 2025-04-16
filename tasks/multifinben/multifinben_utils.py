from comet import download_model, load_from_checkpoint
from datasets import load_dataset


# MT
def comet(items):
    """
    # passthrough for efficiency
    """
    return items


def comet_agg(items):
    """
    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]

    dataset = load_dataset("TheFinAI/DOLFIN_en_es_test")["test"]
    sources = dataset["text"]
    
    data = [
        {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(sources, preds, refs)
    ]

    # Download the model (skip if it already exists)
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    
    # Predict scores (set gpus=1 to use GPU; gpus=0 for CPU)
    model_output = model.predict(data, batch_size=8, gpus=1)
    return model_output.scores[0]
