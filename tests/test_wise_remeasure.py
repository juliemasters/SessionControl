"""Direct native inference must preserve the frozen optimized-training checkpoint."""
import copy
import unittest
from tests.test_wise_qa500 import WISEIntegrationTests
from experiments.wise_qa500.core import export_state,install_state,vendor_modules


class NativeRemeasurementTests(unittest.TestCase):
    def test_frozen_checkpoint_has_identical_direct_native_logits(self):
        torch,native,trained=WISEIntegrationTests().editors()
        official,_=vendor_modules()
        self.assertIs(type(native),official.WISE)
        adapter=trained.get_adapter_layer()
        with torch.no_grad():adapter.new_weight.add_(.15)
        adapter.editing_mean_act.update(.02)
        adapter.memory_weight=[adapter.new_weight.detach().clone()-.05]
        adapter.memory_mean_act=[copy.deepcopy(adapter.editing_mean_act)]
        adapter.editing_total_cnt=508
        state=export_state(trained,500)
        install_state(native,state)
        for ids in [[[3,4,5,6]],[[7,8]],[[9]]]:
            tokens=torch.tensor(ids)
            with torch.inference_mode():
                expected=trained.model(input_ids=tokens).logits
                actual=native.model(input_ids=tokens).logits
            self.assertTrue(torch.equal(expected,actual))
        self.assertEqual(native.get_adapter_layer().editing_total_cnt,508)


if __name__=='__main__':unittest.main()
