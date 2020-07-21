import unittest
import numpy as np
from numpy.testing import assert_allclose
import torch
from VTrace import VTrace
import retro


class RetroKart(unittest.TestCase):
    def test_import(self, game="SuperMarioKart-Snes"):
        """Test that the SuperMarioKart settings were correctly imported"""
        self.assertTrue(game in retro.data.list_games())

    def test_loads(self, game="SuperMarioKart-Snes"):
        """Test that the rom was added with the right version"""
        with self.assertRaises(NameError):
            retro.make(game=game)


class VTraceTest(unittest.TestCase):
    @staticmethod
    def _shaped_arange(*shape):
        """Runs np.arange, converts to float and reshapes."""
        return np.random.randn(np.prod(shape)).astype(np.float32).reshape(
            *shape
        ) * np.prod(shape)

    @staticmethod
    def _ground_truth_calculation(
        discount_factor,
        target_log_policy,
        behaviour_log_policy,
        rewards,
        target_value,
        clip_rho_threshold,
        clip_cs_threshold,
    ):
        """
        Calculates the ground truth for V-trace in Python/Numpy.
        This is borrowed (and partly changed) from the orinal deepmind code
        Values is already bootstrap with next state's prediction
        """
        vs = []
        seq_len = len(target_log_policy)

        importance_sampling = target_log_policy - behaviour_log_policy
        rhos = np.exp(importance_sampling)

        # Truncated importance sampling
        cs = np.minimum(rhos, clip_cs_threshold)
        clipped_rhos = np.minimum(rhos, clip_rho_threshold)

        # Inefficient method close to the iterative formulation
        for s in range(seq_len):
            v_s = np.copy(target_value[s])  # Very important copy.
            for t in range(s, seq_len):
                v_s += (
                    pow(discount_factor, t - s)
                    * np.prod(cs[s:t], axis=0)
                    * clipped_rhos[t]
                    * (
                        rewards[t]
                        + discount_factor * target_value[t + 1]
                        - target_value[t]
                    )
                )
            vs.append(v_s)
        vs = np.stack(vs, axis=0)
        return vs

    def test_vtrace(self, device="cuda"):
        """Tests V-trace against ground truth data calculated in python."""
        seq_len = 5
        batch_size = 6

        # Create log_rhos such that rho will span from near-zero to above the
        # clipping thresholds. In particular, calculate log_rhos in [-2.5, 2.5),
        # so that rho is in approx [0.08, 12.2).
        behaviour_log_rhos = 5 * (
            self._shaped_arange(seq_len, batch_size) / (batch_size * seq_len) - 0.5
        )  # [0.0, 1.0) -> [-2.5, 2.5).
        target_log_rhos = 5 * (
            self._shaped_arange(seq_len, batch_size) / (batch_size * seq_len) - 0.5
        )  # [0.0, 1.0) -> [-2.5, 2.5).

        values = {
            "behaviour_log_policy": behaviour_log_rhos,
            "target_log_policy": target_log_rhos,
            "discount_factor": 0.9,
            "rewards": self._shaped_arange(seq_len, batch_size),
            "target_value": self._shaped_arange(  # We boostrap with t+1
                seq_len + 1, batch_size
            )
            / batch_size,
            "clip_rho_threshold": 3.7,
            "clip_cs_threshold": 1.4,
        }

        vtrace = torch.jit.script(
            VTrace(
                rho=values["clip_rho_threshold"],
                cis=values["clip_cs_threshold"],
                discount_factor=values["discount_factor"],
                sequence_length=seq_len,
            )
        )
        vtrace.to(device)

        output_v, rhos = vtrace(
            target_value=torch.tensor(
                values["target_value"], dtype=float, device=device
            ),
            rewards=torch.tensor(values["rewards"], dtype=float, device=device),
            target_log_policy=torch.tensor(
                values["target_log_policy"], dtype=float, device=device
            ),
            behaviour_log_policy=torch.tensor(
                values["behaviour_log_policy"], dtype=float, device=device
            ),
        )

        vs = self._ground_truth_calculation(**values)

        assert_allclose(vs, output_v.cpu(), rtol=1e-03)


if __name__ == "__main__":
    unittest.main()
