"""CPU checks for activation retrieval and unchanged compact-delta reranking."""
import math
import unittest

import torch

from compact_session_memory import factor_scores
from experiments.session_zsre.core import select_candidate
from experiments.zsre_topk.core import ActivationRouter, normalized_key, routing_summary
from qa500_core import canonical, score as qa500_score


NAMES = ('a.weight',)


def keys(vector):
    return {NAMES[0]: torch.tensor([vector], dtype=torch.float32)}


def payload(right):
    return {NAMES[0]: (torch.tensor([2., -1.]), torch.tensor(right, dtype=torch.float32))}


def example_router():
    # Index row order and bank iteration order intentionally differ. The source
    # activations need not point in the direction of a stored delta's right factor.
    bank = {'z': payload([2., 1.]), 'm': payload([0., 1.]), 'a': payload([1., 0.])}
    matrix = torch.tensor([[0., 1.], [1., 0.], [1. / math.sqrt(2), 1. / math.sqrt(2)]])
    return ActivationRouter(bank, NAMES, matrix, ['z', 'a', 'm']), bank, matrix


class ActivationKeyTests(unittest.TestCase):
    def test_concatenates_in_declared_layer_order_then_normalizes_once(self):
        names = ('earlier.weight', 'later.weight')
        activations = {'later.weight': torch.tensor([[0., 0., 12.]], dtype=torch.float64),
                       'earlier.weight': torch.tensor([[3., 4.]], dtype=torch.float64, requires_grad=True)}
        result = normalized_key(activations, names)
        expected = torch.tensor([3., 4., 0., 0., 12.]) / 13.
        self.assertEqual(result.device.type, 'cpu')
        self.assertEqual(result.dtype, torch.float32)
        self.assertFalse(result.requires_grad)
        self.assertTrue(torch.allclose(result, expected))
        # Per-layer normalization would change the intended relative weighting.
        self.assertAlmostEqual(float(result[0]), 3. / 13., places=7)

    def test_invalid_activations_cannot_silently_enter_the_index(self):
        invalid = [torch.tensor([[0., 0.]]), torch.tensor([[float('nan'), 1.]]),
                   torch.tensor([[float('inf'), 1.]]), torch.tensor([1., 2.]),
                   torch.tensor([[1., 2.], [3., 4.]])]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalized_key({NAMES[0]: value}, NAMES)
        for activations in [{}, {**keys([1., 0.]), 'unexpected.weight': torch.ones(1, 2)}]:
            with self.subTest(names=list(activations)):
                with self.assertRaises(ValueError):
                    normalized_key(activations, NAMES)


class ActivationRouterTests(unittest.TestCase):
    def test_cosine_shortlist_matches_scalar_reference_and_reranks_only_its_members(self):
        router, bank, matrix = example_router()
        query = keys([2., 1.])
        query_vector = [2. / math.sqrt(5), 1. / math.sqrt(5)]
        sids = ['z', 'a', 'm']
        cosine = {sid: sum(float(row[i]) * query_vector[i] for i in range(2))
                  for sid, row in zip(sids, matrix)}
        coarse_order = sorted(sids, key=lambda sid: (-cosine[sid], sid))
        self.assertEqual(coarse_order, ['m', 'a', 'z'])
        result = router.route(query, k=2)
        self.assertEqual(result['coarse_order'], coarse_order)
        self.assertEqual(result['shortlist'], ['m', 'a'])
        candidate_scores = {sid: factor_scores(bank[sid], query)[0] for sid in ['m', 'a']}
        self.assertEqual(result['fine_order'], ['a', 'm'])
        self.assertEqual(result['selected'], 'a')
        self.assertEqual(result['score'], candidate_scores['a'])
        self.assertEqual(result['margin'], candidate_scores['a'] - candidate_scores['m'])
        # The excluded candidate has the largest original fine score. Its score
        # must not override the activation shortlist during reranking.
        self.assertGreater(factor_scores(bank['z'], query)[0], result['score'])
        nearest_only = router.route(query, k=1)
        self.assertEqual(nearest_only['selected'], 'm')
        self.assertEqual(nearest_only['fine_order'], ['m'])
        self.assertIsNone(nearest_only['margin'])

    def test_all_candidates_reproduce_original_selection_scores_margins_and_ranks(self):
        router, bank, _ = example_router()
        query = keys([2., 1.])
        selected, score, margin, ranks = select_candidate(bank, query)
        original_order = sorted(ranks, key=ranks.get)
        full = router.route(query)
        self.assertIsNone(full['shortlist'])
        self.assertIsNone(full['coarse_order'])
        for k in [None, len(bank), len(bank) + 17]:
            with self.subTest(k=k):
                result = router.route(query, k=k)
                self.assertEqual(result['selected'], selected)
                self.assertEqual(result['score'], score)
                self.assertEqual(result['margin'], margin)
                self.assertEqual(result['fine_order'], original_order)
                if k is not None:
                    self.assertEqual(set(result['shortlist']), set(bank))

    def test_fine_score_is_original_multilayer_normalized_delta_energy(self):
        names = ('a.weight', 'b.weight')
        query = {'a.weight': torch.tensor([[2., 1.]]), 'b.weight': torch.tensor([[-1., 3., 2.]])}
        bank = {
            'a': {'a.weight': (torch.tensor([3., 4.]), torch.tensor([2., -1.])),
                  'b.weight': (torch.tensor([1.]), torch.tensor([0., 2., -1.]))},
            'b': {'a.weight': (torch.tensor([4., 0.]), torch.tensor([1., 3.])),
                  'b.weight': (torch.tensor([2.]), torch.tensor([1., 0., 1.]))},
        }
        expected = {}
        for sid, factors in bank.items():
            deltas = {name: left.double()[:, None] * right.double()[None, :]
                      for name, (left, right) in factors.items()}
            numerator = sum(float((delta @ query[name][0].double()).square().sum())
                            for name, delta in deltas.items())
            denominator = sum(float(delta.square().sum()) for delta in deltas.values())
            expected[sid] = numerator / denominator
            self.assertAlmostEqual(factor_scores(factors, query)[0], expected[sid], places=12)
        matrix = torch.stack([normalized_key(query, names)] * len(bank))
        result = ActivationRouter(bank, names, matrix, list(bank)).route(query, k=2)
        self.assertEqual(result['selected'], 'b')
        self.assertEqual(result['fine_order'], ['b', 'a'])
        self.assertAlmostEqual(result['score'], expected['b'], places=12)
        self.assertAlmostEqual(result['margin'], expected['b'] - expected['a'], places=12)

    def test_both_stages_break_exact_ties_by_sid_independent_of_input_order(self):
        bank = {sid: payload([1., 0.]) for sid in ['case_2', 'case_10', 'case_01']}
        sids = ['case_10', 'case_2', 'case_01']
        router = ActivationRouter(bank, NAMES, torch.tensor([[1., 0.]] * 3), sids)
        query = keys([1., 0.])
        expected = ['case_01', 'case_10', 'case_2']
        self.assertEqual(router.route(query, k=2)['coarse_order'], expected)
        self.assertEqual(router.route(query, k=2)['shortlist'], expected[:2])
        for k in [None, 1, 2, 3, 100]:
            with self.subTest(k=k):
                result = router.route(query, k=k)
                self.assertEqual(result['selected'], 'case_01')
                self.assertEqual(result['fine_order'], expected if k is None else expected[:min(k, 3)])
                if k != 1:
                    self.assertEqual(result['margin'], 0.)

    def test_unique_stored_query_retrieves_itself_without_case_or_answer_input(self):
        router, _, matrix = example_router()
        for sid, vector in zip(['z', 'a', 'm'], matrix):
            with self.subTest(sid=sid):
                result = router.route(keys(vector.tolist()), k=1)
                self.assertEqual(result['shortlist'], [sid])
                self.assertEqual(result['selected'], sid)
        for forbidden in [{'case_id': 3}, {'answer': 'held-out answer'}, {'target': 'reference'}]:
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(TypeError):
                    router.route(keys([2., 1.]), k=1, **forbidden)

    def test_invalid_k_and_query_width_are_rejected(self):
        router, _, _ = example_router()
        for k in [0, -1, True, False, 1.5, '1']:
            with self.subTest(k=k):
                with self.assertRaises((ValueError, TypeError)):
                    router.route(keys([2., 1.]), k=k)
        with self.assertRaises(ValueError):
            router.route(keys([1., 2., 3.]), k=1)

    def test_index_validation_prevents_ambiguous_or_corrupt_candidate_mapping(self):
        bank = {'a': payload([1., 0.]), 'b': payload([0., 1.])}
        valid = torch.eye(2)
        invalid = [(valid, ['a', 'a']), (valid, ['a', 'missing']), (valid, ['a']),
                   (torch.ones(2), ['a', 'b']), (torch.eye(3), ['a', 'b']),
                   (valid.double(), ['a', 'b']), (valid * 2, ['a', 'b']),
                   (torch.zeros(2, 2), ['a', 'b']),
                   (torch.tensor([[float('nan'), 0.], [0., 1.]]), ['a', 'b'])]
        for matrix, sids in invalid:
            with self.subTest(shape=tuple(matrix.shape), dtype=matrix.dtype, sids=sids):
                with self.assertRaises(ValueError):
                    ActivationRouter(bank, NAMES, matrix, sids)

    def test_full_route_does_not_require_a_coarse_index(self):
        _, bank, _ = example_router()
        query = keys([2., 1.])
        router = ActivationRouter(bank, NAMES, None, list(reversed(bank)))
        selected, score, margin, _ = select_candidate(bank, query)
        result = router.route(query)
        self.assertEqual((result['selected'], result['score'], result['margin']), (selected, score, margin))
        with self.assertRaises(ValueError):
            router.route(query, k=1)

    def test_multiple_question_keys_use_max_similarity_and_k_distinct_deltas(self):
        bank = {'c': payload([1., 1.]), 'a': payload([0., 1.]), 'b': payload([1., 0.])}
        sids = ['b', 'c', 'a']
        matrix = torch.tensor([[1., 0.], [0., 1.], [.95, math.sqrt(1. - .95 ** 2)],
                               [.8, .6], [-1., 0.]])
        index_sids = ['a', 'c', 'a', 'b', 'a']
        scored = []

        def tracked_score(factors, query):
            scored.append(next(sid for sid in bank if bank[sid] is factors))
            return factor_scores(factors, query)[0]

        router = ActivationRouter(bank, NAMES, matrix, sids,
                                  index_sids=index_sids, scorer=tracked_score)
        result = router.route(keys([1., 0.]), k=2)
        # Two best key rows both belong to a. They must consume only one slot.
        # a's opposite key also makes its average lower than b's similarity,
        # so this ordering specifically requires maximum rather than mean pooling.
        self.assertEqual(result['coarse_order'], ['a', 'b', 'c'])
        self.assertEqual(result['shortlist'], ['a', 'b'])
        self.assertEqual(scored, ['a', 'b'])
        self.assertEqual(result['selected'], 'b')
        self.assertEqual(result['fine_order'], ['b', 'a'])
        self.assertEqual(router.route(keys([1., 0.]), k=10)['shortlist'], ['a', 'b', 'c'])
        for invalid_sids in [['a', 'c', 'a', 'b'], ['a', 'a', 'a', 'b', 'a'],
                             ['a', 'c', 'a', 'unknown', 'a']]:
            with self.subTest(index_sids=invalid_sids):
                with self.assertRaises(ValueError):
                    ActivationRouter(bank, NAMES, matrix, sids, index_sids=invalid_sids)

    def test_canonical_rank_two_fine_scores_match_native_qa500_and_dense_deltas(self):
        names = ('a.weight', 'b.weight')
        query = {'a.weight': torch.tensor([[2., -1., 3.]]), 'b.weight': torch.tensor([[4., 1.]])}
        bank = {
            'two': canonical({
                'a.weight': (torch.tensor([[1., 2.], [3., 1.]]),
                             torch.tensor([[1., 0.], [0., 2.], [1., 1.]])),
                'b.weight': (torch.tensor([[2., 1.], [0., 1.], [1., -1.]]),
                             torch.tensor([[1., 2.], [-1., 1.]])),
            }),
            'other': canonical({
                'a.weight': (torch.tensor([[2., 1.], [-1., 2.]]),
                             torch.tensor([[1., 1.], [2., -1.], [0., 2.]])),
                'b.weight': (torch.tensor([[1., 1.], [2., 0.], [1., 2.]]),
                             torch.tensor([[2., 1.], [1., -1.]])),
            }),
            'one': canonical({
                'a.weight': (torch.tensor([1., 2.]), torch.tensor([1., -1., 2.])),
                'b.weight': (torch.tensor([2., 1., 1.]), torch.tensor([1., 2.])),
            }),
        }
        dense_scores = {}
        for sid, factors in bank.items():
            deltas = {name: left.double() @ right.double().T
                      for name, (left, right) in factors.items()}
            numerator = sum(float((delta @ query[name][0].double()).square().sum())
                            for name, delta in deltas.items())
            denominator = sum(float(delta.square().sum()) for delta in deltas.values())
            dense_scores[sid] = numerator / denominator
            self.assertAlmostEqual(qa500_score(factors, query), dense_scores[sid], places=12)
        expected_order = sorted(bank, key=lambda sid: (-dense_scores[sid], sid))
        native_scores = {sid: qa500_score(factors, query) for sid, factors in bank.items()}
        matrix = torch.stack([normalized_key(query, names)] * len(bank))
        router = ActivationRouter(bank, names, matrix, list(reversed(bank)), scorer=qa500_score)
        for k in [None, len(bank), len(bank) + 9]:
            with self.subTest(k=k):
                result = router.route(query, k=k)
                self.assertEqual(result['fine_order'], expected_order)
                self.assertEqual(result['selected'], expected_order[0])
                self.assertEqual(result['score'], native_scores[expected_order[0]])
                self.assertEqual(result['margin'], native_scores[expected_order[0]] - native_scores[expected_order[1]])


class RoutingSummaryTests(unittest.TestCase):
    def test_multi_support_hits_and_abstentions_have_the_correct_denominators(self):
        support = {'a', 'b'}
        decisions = [('b', ['c', 'b']), ('a', ['a', 'c']),
                     ('c', ['b', 'c']), ('c', ['c', 'd'])]
        trials = [dict(selected=selected, source_hit=selected in support,
                       shortlist_hit=bool(support.intersection(shortlist)))
                  for selected, shortlist in decisions]
        # Abstention items are still questions, but have no source-routing label.
        trials.extend([dict(selected='a', source_hit=None, shortlist_hit=None),
                       dict(selected='c', source_hit=None, shortlist_hit=None)])
        result = routing_summary(trials)
        self.assertEqual(result['questions'], 6)
        self.assertEqual(result['routing_questions'], 4)
        self.assertEqual(result['correct'], 2)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['shortlist_correct'], 3)
        self.assertEqual(result['shortlist_recall'], .75)

    def test_all_abstentions_and_full_scan_do_not_invent_shortlist_measurements(self):
        abstentions = [dict(source_hit=None, shortlist_hit=None)] * 2
        result = routing_summary(abstentions)
        self.assertEqual((result['questions'], result['routing_questions'], result['correct']), (2, 0, 0))
        self.assertIsNone(result['accuracy'])
        self.assertIsNone(result['shortlist_recall'])
        self.assertIsNone(result['shortlist_correct'])
        full_scan = routing_summary([dict(source_hit=True, shortlist_hit=None),
                                     dict(source_hit=False, shortlist_hit=None), *abstentions])
        self.assertEqual(full_scan['accuracy'], .5)
        self.assertEqual(full_scan['routing_questions'], 2)
        self.assertIsNone(full_scan['shortlist_recall'])
        self.assertIsNone(full_scan['shortlist_correct'])


if __name__ == '__main__':
    unittest.main()
