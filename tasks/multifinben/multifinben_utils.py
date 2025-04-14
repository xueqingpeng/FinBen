from comet import download_model, load_from_checkpoint

def get_comet_score(doc, results):
    """
    Calculate the quality score of machine translation using the COMET model.
    """
    # Download the model (skip if it already exists)
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    source = doc["text"]
    mt = results[0].strip()
    ref = doc["answer"]

    data = [
        {
            "src": source, # original text
            "mt": mt,  # machine translation
            "ref": ref # human reference translation
        },
    ]

    # Predict scores (set gpus=1 to use GPU; gpus=0 for CPU)
    model_output = model.predict(data, batch_size=8, gpus=1)
    return {"COMET": model_output.scores[0]}
