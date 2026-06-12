import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LORA_SOURCE = ROOT / "finetune_lora.py"
MAIN_SOURCE = ROOT / "main.py"
ENGINE_SOURCE = ROOT / "robustft" / "engine.py"


def function_node(name: str) -> ast.FunctionDef:
    tree = ast.parse(LORA_SOURCE.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


class LoraPipelineContractTests(unittest.TestCase):
    def test_prepare_targets_accepts_training_indices(self):
        args = [arg.arg for arg in function_node("prepare_targets").args.args]
        self.assertIn("train_idx", args)

    def test_prediction_exposes_explicit_checkpoint_policy(self):
        source = LORA_SOURCE.read_text(encoding="utf-8")
        self.assertIn('--checkpoint', source)
        self.assertNotIn('"best.pt" if', source)

    def test_full_training_saves_full_checkpoint(self):
        source = ast.unparse(function_node("train"))
        self.assertIn("full.pt", source)

    def test_feature_cache_records_extraction_recipe(self):
        source = MAIN_SOURCE.read_text(encoding="utf-8")
        self.assertIn('"model_name": MODEL_NAME', source)
        self.assertIn('"tta_flip":', source)

    def test_lora_training_uses_compact_targets(self):
        source = LORA_SOURCE.read_text(encoding="utf-8")
        self.assertIn("target_labels_all", source)
        self.assertNotIn("targets_all =", source)

    def test_checkpoints_include_resumable_training_state(self):
        source = LORA_SOURCE.read_text(encoding="utf-8")
        self.assertIn("--resume", source)
        self.assertIn('"optimizer": opt.state_dict()', source)
        self.assertIn('"scheduler": sched.state_dict()', source)

    def test_stratified_split_preserves_one_training_sample_per_class(self):
        source = ENGINE_SOURCE.read_text(encoding="utf-8")
        self.assertIn("len(idx) - 1", source)


if __name__ == "__main__":
    unittest.main()
