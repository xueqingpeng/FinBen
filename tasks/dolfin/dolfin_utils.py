from comet import download_model, load_from_checkpoint
from datasets import load_dataset


# MT
def sliding_window(srcs, mts, refs, window_size=3, stride=1):
    data = []
    for i in range(0, len(srcs) - window_size + 1, stride):
        src = " ".join(srcs[i:i+window_size])
        mt = " ".join(mts[i:i+window_size])
        ref = " ".join(refs[i:i+window_size])
        data.append({"src": src, "mt": mt, "ref": ref})
    return data


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
    
    # data = [
    #     {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(sources, preds, refs)
    # ]

    # Ensure alignment
    assert len(sources) == len(refs) == len(preds), "Length mismatch among sources, refs, preds"

    # Apply SLIDE approach
    data = sliding_window(sources, preds, refs, window_size=3, stride=1)

    # Download the model (skip if it already exists)
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    
    # Predict scores (set gpus=1 to use GPU; gpus=0 for CPU)
    model_output = model.predict(data, batch_size=8, gpus=1)
    return sum(model_output.scores) / len(model_output.scores)
