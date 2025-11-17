from transformers import AutoTokenizer, AutoModel

def get_text_transformer_model(name, attn_implementation="spda"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, attn_implementation=attn_implementation)
    return tokenizer, model

def get_vision_transformer_model(name, attn_implementation="spda"):
    return AutoModel.from_pretrained(name, attn_implementation=attn_implementation)