import ast

def family_tag_from_source(source: str) -> str:
    s = source.lower()
    if "fractalblock" in s or "fractalunit" in s or "fractal_fn" in s:
        return "fractal"
    if "transformerencoderlayer" in s or "multiheadattention" in s:
        return "transformer"
    if "depthwise" in s or "mobilenet" in s:
        return "mobile"
    if "resnet" in s or "residual" in s:
        return "resnet"
    if "densenet" in s or "dense" in s:
        return "densenet"
    if "vgg" in s:
        return "vgg"
    return "generic"

def get_model_family(name: str, code: str) -> str:
    # Prefer name hints
    if name:
        for prefix in ['Fractal', 'ResNet', 'VGG', 'DenseNet', 'EfficientNet', 'MobileNet',
                       'Transformer', 'BERT', 'GPT', 'RAG', 'Rag', 'rag']:
            if name.startswith(prefix):
                if prefix.lower().startswith("fractal"):
                    return "fractal"
                if prefix.lower().startswith("resnet"):
                    return "resnet"
                if prefix.lower().startswith("vgg"):
                    return "vgg"
                if prefix.lower().startswith("dense"):
                    return "densenet"
                if prefix.lower().startswith("mobile"):
                    return "mobile"
                if prefix.lower().startswith("transformer") or prefix.lower() in ("bert","gpt"):
                    return "transformer"
                if prefix.lower().startswith("rag"):
                    return "generic"
    # Fallback to code cues
    return family_tag_from_source(code)
