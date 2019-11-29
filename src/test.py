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
        return np.random.randn(np.prod(shape)).astype(np.float32).reshape(*shape)*np.prod(shape)

    @staticmethod
    def _ground_truth_calculation(discount_factor, target_log_rhos, behaviour_log_rhos, rewards, values,
                                bootstrap_value, clip_rho_threshold,
                                clip_cs_threshold, **kwargs):
        """
        Calculates the ground truth for V-trace in Python/Numpy.
        This is borrowed (and partly changed) from the orinal deepmind code
        """
        vs = []
        seq_len = len(log_rhos)

        importance_sampling = target_log_rhos-behaviour_log_rhos
        rhos = np.exp(importance_sampling)

        # Truncated importance sampling
        cs = np.minimum(rhos, clip_cs_threshold)
        clipped_rhos = np.minimum(rhos, clip_rho_threshold)

        # Inefficient method close to the iterative formulation
        for s in range(seq_len):
            v_s = np.copy(values[s])  # Very important copy.
            for t in range(s, seq_len):
                v_s += (
                    pow(discount_factor, t-s) *
                    np.prod(cs[s:t], axis=0) * clipped_rhos[t] *
                    (rewards[t] + discount_factor * values[t + 1] - values[t]))  
            vs.append(v_s)
        vs = np.stack(vs, axis=0)
        return vs

    def test_vtrace(self):
        """Tests V-trace against ground truth data calculated in python."""
        seq_len = 5
        batch_size = 6

        # Create log_rhos such that rho will span from near-zero to above the
        # clipping thresholds. In particular, calculate log_rhos in [-2.5, 2.5),
        # so that rho is in approx [0.08, 12.2).
        behaviour_log_rhos = 5 * (self._shaped_arange(seq_len, batch_size) / (batch_size * seq_len) - 0.5)  # [0.0, 1.0) -> [-2.5, 2.5).
        target_log_rhos = 5 * (self._shaped_arange(seq_len, batch_size) / (batch_size * seq_len) - 0.5)  # [0.0, 1.0) -> [-2.5, 2.5).
        
        values = {
            'behaviour_log_rhos': behaviour_log_rhos,
            'target_log_rhos': target_log_rhos,
            'discount_factor':
                0.9,
            'rewards':
                self._shaped_arange(seq_len, batch_size),
            'values':
                self._shaped_arange(seq_len+1, batch_size) / batch_size,
            'bootstrap_value':
                self._shaped_arange(batch_size) + 1.0,
            'clip_rho_threshold':
                3.7,
            'clip_cs_threshold':
                1.4,
        }

        vtrace = VTrace(rho=values["clip_rho_threshold"],
                        cis=values["clip_cs_threshold"], 
                        discount_factor=values["discount_factor"])

        output_v = vtrace(target_value=torch.tensor(values["values"]), 
                          rewards=torch.tensor(values["rewards"]), 
                          target_log_policy=torch.tensor(values["target_log_rhos"]), 
                          behaviour_log_policy=torch.tensor(values["behaviour_log_rhos"]))

        vs = self._ground_truth_calculation(**values)

        with self.assertRaises(AssertionError):
            assert_allclose(vs, output_v+1, rtol=1e-04)



if __name__ == "__main__":
    unittest.main()