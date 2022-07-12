import unittest
import torch
from src.models.gMLP import model_getter
from src.models.GPT2 import model_getter as model_getter_gpt2


def get_zero_grad_hook(mask):
    def hook(grad):
        return grad * mask

    return hook


class TestTriangular(unittest.TestCase):
    """
    Tests basic gMLP functionality around the causal mixing + masking
    """

    def setUp(self) -> None:

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        self.device = device

        self.seq_len = 128
        self.model = model_getter(
            "qa",
            vocab_size=50257,
            num_ctx=self.seq_len,
            **{"tied_head": True, "tiny_attn": True}
        )

        self.model.to(self.device)
        self.model.prepare()

    def tearDown(self) -> None:
        del self.model

    def test_triangular_hook(self):
        """
        Ensure triangular weights at model initialization
        """

        eq_cond = torch.sum(
            torch.eq(
                self.model.gmlpblocks[0].sgu.token_mix.fc_causal.weight,
                torch.tril(
                    self.model.gmlpblocks[0].sgu.token_mix.fc_causal.weight
                ),
            )
        )

        self.assertEqual(eq_cond, self.seq_len ** 2)

    def test_triangular_persists(self):
        """
        Ensure the gradient hook persists across multiple training steps
        """

        self.model.to(self.device)
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=3e-4, weight_decay=0.01
        )

        for _ in range(0, 250):
            data_batch = torch.randint(
                low=0, high=50257, size=(4, self.seq_len)
            ).to(self.device)

            optimizer.zero_grad()

            logits, loss = self.model(data_batch, data_batch)

            loss.backward()

            optimizer.step()

        eq_cond = torch.sum(
            torch.eq(
                self.model.gmlpblocks[0].sgu.token_mix.fc_causal.weight,
                torch.tril(
                    self.model.gmlpblocks[0].sgu.token_mix.fc_causal.weight
                ),
            )
        )

        self.assertEqual(eq_cond, self.seq_len ** 2)


class TestGPT(unittest.TestCase):
    """
    Testing basic transformer functionality
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

        self.model.cuda()

    def tearDown(self) -> None:
        del self.model

    def test_forward(self):
        data_batch = torch.randint(
            low=0, high=50257, size=(4, self.seq_len)
        ).cuda()

        # forward with labels
        logits, loss = self.model(data_batch, data_batch)

        # forward with no labels
        logits = self.model(data_batch)


class TestALiBi(unittest.TestCase):
    """
    Testing basic ALiBi functionality
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
            **{"fused_residuals": False, "num_head": 4, "use_alibi": True}
        )

        self.model.to(self.device)

    def tearDown(self) -> None:
        del self.model

    def test_forward(self):
        data_batch = torch.randint(
            low=0, high=50257, size=(4, self.seq_len)
        ).to(self.device)

        # forward with labels
        logits, loss = self.model(data_batch, data_batch)

        # forward with no labels
        logits = self.model(data_batch)

    def test_change_ctx(self):
        # Double sequence length, make sure no errors
        data_batch = torch.randint(
            low=0, high=50257, size=(4, 2 * self.seq_len)
        ).to(self.device)

        with torch.no_grad():
            logits, loss = self.model(data_batch, data_batch)

        # halve sequence length to original, make sure no errors

        data_batch = torch.randint(
            low=0, high=50257, size=(4, self.seq_len)
        ).to(self.device)

        with torch.no_grad():
            logits, loss = self.model(data_batch, data_batch)
