import torch
import sentencepiece as spm
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config, load_checkpoint
from src.inference.generate import generate

def test_factual_completions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    prompts = [
        "Canberra is the capital of",
        "Au is the chemical symbol for",
        "Neil Armstrong walked on the",
        "The first president of the United States was",
        "Dmitri Mendeleev created the",
        "Photosynthesis converts sunlight into",
        "World War II ended in",
    ]
    
    expected = [
        "Australia",
        "gold",
        "moon",
        "Washington",
        "periodic table",
        "energy",
        "1945",
    ]
    
    print("Loading models...")
    cfg_1_0 = load_config("configs/finetune_1.0.yaml")
    model_cfg_1_0 = ModelConfig.from_dict(cfg_1_0["model"])
    model_1_0 = Transformer(model_cfg_1_0).to(device)
    load_checkpoint("checkpoints/finetune_1.0_final.pt", model_1_0)
    tokenizer_1_0 = spm.SentencePieceProcessor()
    tokenizer_1_0.load("data/tokenizer_1.0.model")
    
    cfg_1_1 = load_config("configs/pretrain_1.1.yaml")
    model_cfg_1_1 = ModelConfig.from_dict(cfg_1_1["model"])
    model_1_1 = Transformer(model_cfg_1_1).to(device)
    load_checkpoint("checkpoints/1.1_step_210000.pt", model_1_1)
    tokenizer_1_1 = spm.SentencePieceProcessor()
    tokenizer_1_1.load("data/tokenizer_1.1.model")
    print("Models loaded.\n")
    
    print("FACTUAL COMPLETION TEST (temp=0.0, greedy)")
    print("=" * 80)
    print()
    
    results = []
    for i, (prompt, exp) in enumerate(zip(prompts, expected), 1):
        print(f"Test {i}: {prompt}")
        print(f"Expected: {exp}")
        
        # Full outputs for context
        out_1_0 = generate(
            model_1_0, tokenizer_1_0, prompt,
            max_tokens=20,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
            device=device
        )
        
        out_1_1 = generate(
            model_1_1, tokenizer_1_1, prompt,
            max_tokens=20,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
            device=device
        )
        
        comp_1_0 = out_1_0[len(prompt):].strip()
        comp_1_1 = out_1_1[len(prompt):].strip()
        
        knows_1_0 = any(exp.lower() in word.lower() for word in comp_1_0.split()[:3])
        knows_1_1 = any(exp.lower() in word.lower() for word in comp_1_1.split()[:3])
        
        print(f"1.0 (finetune): {comp_1_0} {'[CORRECT]' if knows_1_0 else '[WRONG]'}")
        print(f"1.1 (pretrain): {comp_1_1} {'[CORRECT]' if knows_1_1 else '[WRONG]'}")
        print()
        
        results.append((knows_1_0, knows_1_1))
    
    print("=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    score_1_0 = sum(r[0] for r in results)
    score_1_1 = sum(r[1] for r in results)
    gaps = sum(1 for r in results if r[0] and not r[1])
    regressions = sum(1 for r in results if not r[0] and r[1])
    
    print(f"1.0 (finetune) correct: {score_1_0}/7")
    print(f"1.1 (pretrain) correct: {score_1_1}/7")
    print(f"Gaps (1.0 > 1.1):      {gaps}")
    print(f"Improvements (1.1 > 1.0): {regressions}")
    
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    if gaps == 0:
        verdict = "NO REGRESSION"
        msg = f"1.0 knows {score_1_0}, 1.1 knows {score_1_1}. No facts lost."
        cause = "SFT likely enables 'answer mode' in 1.0, not base knowledge difference."
    else:
        verdict = "KNOWLEDGE GAP"
        msg = f"{gaps} facts 1.0 knows but 1.1 doesn't."
        cause = "Either SFT effect or actual 1.1 base model limitation."
    
    print(f"{verdict}: {msg}")
    print(f"Likely cause: {cause}")

if __name__ == "__main__":
    test_factual_completions()
