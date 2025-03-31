from huggingface_hub import HfApi
import os
import json
import shutil
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def upload_model_to_hf_hub(
    local_model_path,
    hf_model_id,
    hf_token=None
):
    """
    Upload a local model to Hugging Face Hub

    Args:
        :param local_model_path:
        :param hf_model_id:
        :param hf_token:
        :return:
    """
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    # Load label mappings
    label_mappings_path = os.path.join(local_model_path, 'label_mappings.json')
    if os.path.exists(label_mappings_path):
        with open(label_mappings_path, 'r') as f:
            mappings = json.load(f)
            label2id = mappings['label2id']
            id2label = {int(k): v for k, v in mappings['id2label'].items()}

        # Set the mappings in the model config
        model.config.label2id = label2id
        model.config.id2label = id2label

    # Push to hub
    model.push_to_hub(hf_model_id, use_auth_token=hf_token)
    tokenizer.push_to_hub(hf_model_id, use_auth_token=hf_token)

    print(f"Model successfully uploaded to {hf_model_id}")
    print(f"You can now use it with: model = AutoModelForSequenceClassification.from_pretrained('{hf_model_id}')")

if __name__ == "__main__":
    local_model_path = "./model_export"
    hf_model_id = "AfroLogicInsect/emotionClassifier"

    upload_model_to_hf_hub(local_model_path, hf_model_id)

