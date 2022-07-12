import unittest
import torch
from src.models.GPT2 import model_getter as model_getter_gpt2
from src.utils.adapters import add_adapters, prepare_adapter_training


class TestAdapterBasic(unittest.TestCase):
    """
    Test basic creation of adapters
    """

    def setUp(self) -> None:

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        self.device = device

        self.seq_len = 128
        self.model = model_getter_gpt2(
            "qa",
            vocab_size=50257,
            num_ctx=self.seq_len,
            **{"fused_residuals": False, "num_head": 4, "use_alibi": False}
        )

        self.model.to(self.device)

    def tearDown(self) -> None:
        del self.model

    def test_add_adapter(self):
        # Ensure adding adapters works
        adapter = add_adapters(self.model, reduction_factor=8)
        adapter.to(self.device)
        test_batch = torch.randint(
            low=0, high=50257, size=(4, self.seq_len)
        ).to(self.device)
        adapter(test_batch)

    def test_adapter_training(self):
        # Ensure we can disable gradients for adapter training (check that after the prepare function is called we have less params)
        adapter = add_adapters(self.model, reduction_factor=8)
        all_trainable_params = sum(
            p.numel() for p in adapter.parameters() if p.requires_grad == True
        )
        adapter = prepare_adapter_training(adapter)
        test_batch = torch.randint(
            low=0, high=50257, size=(4, self.seq_len)
        ).to(self.device)
        adapter.to(self.device)
        adapter(test_batch)

        adapter_trainable_params = sum(
            p.numel() for p in adapter.parameters() if p.requires_grad == True
        )

        self.assertGreater(all_trainable_params, adapter_trainable_params)

    def test_adapter_fused_residuals(self):
        model = model_getter_gpt2(
            "qa",
            vocab_size=50257,
            num_ctx=self.seq_len,
            **{"fused_residuals": True, "num_head": 4, "use_alibi": False}
        )

        model.to(self.device)

        adapter = add_adapters(self.model, reduction_factor=8)
        adapter = prepare_adapter_training(adapter)
        test_batch = torch.randint(
            low=0, high=50257, size=(4, self.seq_len)
        ).to(self.device)
        adapter.to(self.device)
        adapter(test_batch)

    def test_adapter_alibi(self):
        model = model_getter_gpt2(
            "qa",
            vocab_size=50257,
            num_ctx=self.seq_len,
            **{"fused_residuals": False, "num_head": 4, "use_alibi": True}
        )

        model.to(self.device)
        adapter = add_adapters(self.model, reduction_factor=8)
        adapter = prepare_adapter_training(adapter)
        test_batch = torch.randint(
            low=0, high=50257, size=(4, self.seq_len)
        ).to(self.device)
        adapter.to(self.device)
        adapter(test_batch)
