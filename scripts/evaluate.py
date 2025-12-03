import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from transformers import BertTokenizer
from src.config import ModelConfig, CULTURAL_THEMES
from src.models.prompt_tuning_model import CulturalPromptTuning
from src.training.evaluator import CulturalEvaluator


def main():
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = ModelConfig()
    model = CulturalPromptTuning(model_config).to(device)
    model.load_state_dict(
        torch.load("models/final_model_no_early_stop.pt", map_location=device)
    )
    model.eval()

    # Initialize evaluator
    id_to_theme = {v: k for k, v in CULTURAL_THEMES.items()}
    evaluator = CulturalEvaluator(model, device, id_to_theme)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Test examples
    examples = [
        "PersonX visits elderly relatives with gifts after Eid prayer.",
        "PersonX prepares salted fish and colored eggs for family during Sham El Nessim.",
    ]

    print("\n🔍 TOP-2 CULTURAL THEME PREDICTIONS")
    for example in examples:
        top2 = evaluator.predict_top2(example, tokenizer)
        print(f"\nText: {example[:75]}{'...' if len(example) > 75 else ''}")
        for i, pred in enumerate(top2, 1):
            print(
                f"  {i}. {pred['theme']:<25} (ID: {pred['id']}) | Confidence: {pred['confidence']:.3f}"
            )


if __name__ == "__main__":
    main()
