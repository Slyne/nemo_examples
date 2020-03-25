"""External Scorer for Beam Search Decoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import kenlm
import numpy as np


class Scorer(object):
    """External scorer to evaluate a prefix or whole sentence in
       beam search decoding, including the score from n-gram language
       model and word count.

    :param alpha: Parameter associated with language model. Don't use
                  language model when alpha = 0.
    :type alpha: float
    :param beta: Parameter associated with word count. Don't use word
                count when beta = 0.
    :type beta: float
    :model_path: Path to load language model.
    :type model_path: basestring
    """

    def __init__(self, alpha, beta, model_path):
        self._alpha = alpha
        self._beta = beta
        if not os.path.isfile(model_path):
            raise IOError("Invaid language model path: %s" % model_path)
        self._language_model = kenlm.LanguageModel(model_path)

    # n-gram language model scoring
    def _language_model_score(self, sent_lastword, state):
        #log10 prob of last word
        state2 = kenlm.State()
        log_cond_prob = self._language_model.BaseScore(state, sent_lastword, state2)
        return np.power(10, log_cond_prob), state2

    # reset alpha and beta
    def reset_params(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta

    # execute evaluation
    def __call__(self, sent_lastword, state,word_cnt, log=False):
        """Evaluation function, gathering all the different scores
        and return the final one.

        :param sentence: The input sentence for evalutation
        :type sentence: basestring
        :param log: Whether return the score in log representation.
        :type log: bool
        :return: Evaluation score, in the decimal or log.
        :rtype: float
        """
        cur_state = state
        if word_cnt == 0:
            if cur_state is None:
                state = kenlm.State()
            cur_state = self._language_model.BeginSentenceWrite(state)

        lm, new_state = self._language_model_score(sent_lastword, cur_state)
        if log == False:
            score = np.power(lm, self._alpha) * np.power(word_cnt, self._beta)
        else:
            score = self._alpha * np.log(lm) + self._beta * np.log(word_cnt)
        return score, new_state

