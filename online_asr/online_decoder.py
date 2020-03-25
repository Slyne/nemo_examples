import numpy as np

class OnlineDecoder(object):

    def __init__(self, blank_id, vocabulary, decoder_type, ext_scoring_func=None, char_based_lm=False):
        self.blank_id = blank_id
        self.vocabulary = vocabulary  # must contain blank id
        self.decoder_type = decoder_type
        self.ext_scoring_func = ext_scoring_func
        self.decoder_type = decoder_type
        self.reset(decoder_type)   
        self.char_based_lm = char_based_lm

    def reset(self,decoder_type):
        if decoder_type == "ctc_beam_search":
            self.reset_beam_search_decoder()
        elif decoder_type == "greedy_search":
            self.reset_greedy_search_decoder()

    def decode(self, logits, **kwargs): 
        if self.decoder_type == "ctc_beam_search":
            return self.ctc_beam_decode_chunk(logits, **kwargs)
        else:
            return self.ctc_greedy_decode_chunk(logits, **kwargs)

    def reset_beam_search_decoder(self):
        self.state = None
        self.wordcount = 0
        self.prefix_set_prev = {'\t': (0.0, self.state, '', self.wordcount)}
        self.log_probs_b_prev, self.log_probs_nb_prev = {'\t': 0.0}, {'\t': -float("INF")}
    
    def reset_greedy_search_decoder(self):
        self.prev_char = ''
    
    def ctc_greedy_decode_chunk(self, probs_seq, merge=True):
        merged = ''
        unmerged = ''
        
        for logit in probs_seq:
            cur_char = self.vocabulary[np.argmax(logit)]
            if not merge:
                unmerged += cur_char
            if cur_char != self.prev_char:
                self.prev_char = cur_char
                if cur_char != self.vocabulary[self.blank_id]:
                    merged += cur_char
        return merged if merge else unmerged
        
    def ctc_beam_decode_chunk(self, log_probs_seq, cutoff_prob=0.99, cutoff_top_n=10, beam_size=20, **kwargs):
        ## extend prefix in loop
        for logit in log_probs_seq:   
           # turn logit to prob  
            if not len(logit) == len(self.vocabulary):
                raise ValueError("The shape of prob_seq does not match with the "
                                "shape of the vocabulary.")
            # prefix_set_next: the set containing candidate prefixes
            # probs_b_cur: prefixes' probability ending with blank in current step
            # probs_nb_cur: prefixes' probability ending with non-blank in current step
            prefix_set_next, log_probs_b_cur, log_probs_nb_cur = {}, {}, {}

            log_prob_idx = self._get_pruned_log_probs(logit, cutoff_prob, cutoff_top_n)

            for l in self.prefix_set_prev:
                if l not in prefix_set_next:
                    log_probs_b_cur[l], log_probs_nb_cur[l] = -float("INF"), -float("INF")

                _, org_state, org_word, org_count = self.prefix_set_prev[l]
                
                # extend prefix by travering prob_idx
                for c, log_prob_c in log_prob_idx:
                    cur_prefix_state, cur_last_word, cur_word_count = (org_state, org_word, org_count)
                   
                    if c == self.blank_id:
                        log_probs_b_cur[l] = np.logaddexp(log_probs_b_cur[l], \
                                            log_prob_c + np.logaddexp(self.log_probs_b_prev[l],\
                                            self.log_probs_nb_prev[l]))
                    else:
                        last_char = l[-1]
                        new_char = self.vocabulary[c]
                        l_plus = l + new_char
                        if l_plus not in prefix_set_next:
                            log_probs_b_cur[l_plus], log_probs_nb_cur[l_plus] = -float("INF"), -float("INF")

                        if new_char == last_char:
                            log_probs_nb_cur[l_plus] = np.logaddexp(log_probs_nb_cur[l_plus], log_prob_c + self.log_probs_b_prev[l])
                            log_probs_nb_cur[l] = np.logaddexp(log_probs_nb_cur[l],log_prob_c +  self.log_probs_nb_prev[l])
                        elif new_char == ' ' or self.char_based_lm:
                            if cur_last_word != '':
                                    #print("**"+cur_last_word+"**")
                                    cur_word_count = cur_word_count + 1

                            if (self.ext_scoring_func is None) or (len(l) == 1):
                                log_score = 0.0
                            else:
                                log_score, cur_prefix_state = \
                                        self.ext_scoring_func(cur_last_word, cur_prefix_state, cur_word_count, log=True)
                            
                            if not self.char_based_lm:
                                cur_last_word = ''
                           
                            log_probs_nb_cur[l_plus] = np.logaddexp(log_probs_nb_cur[l_plus], log_score + log_prob_c +
                                np.logaddexp(self.log_probs_b_prev[l], self.log_probs_nb_prev[l]))
                        else: 
                            cur_last_word = cur_last_word + new_char                       
                            log_probs_nb_cur[l_plus] = np.logaddexp(log_probs_nb_cur[l_plus], log_prob_c +
                                np.logaddexp(self.log_probs_b_prev[l], self.log_probs_nb_prev[l]))
                        
                        if self.char_based_lm:
                            cur_last_word = new_char
                        # add l_plus into prefix_set_next
                        prefix_set_next[l_plus] = (np.logaddexp(log_probs_nb_cur[
                            l_plus],log_probs_b_cur[l_plus]), cur_prefix_state, cur_last_word, cur_word_count)
                # add l into prefix_set_next
        
                prefix_set_next[l]= (np.logaddexp(log_probs_b_cur[l], log_probs_nb_cur[l]), org_state, org_word, org_count)
            # update probs
            self.log_probs_b_prev, self.log_probs_nb_prev = log_probs_b_cur, log_probs_nb_cur

            ## store top beam_size prefixes
            self.prefix_set_prev = sorted(
                prefix_set_next.items(), key=lambda asd: asd[1][0], reverse=True)
            if beam_size < len(self.prefix_set_prev):
                self.prefix_set_prev = self.prefix_set_prev[:beam_size]
            self.prefix_set_prev = dict(self.prefix_set_prev)
        
        beam_result = []
        for seq, cur_total_state in self.prefix_set_prev.items():
            log_prob, state, last_word, word_count = cur_total_state
            if log_prob > float("-INF") and len(seq) > 1:
                # score last word by external scorer
                if (self.ext_scoring_func is not None) and (last_word != ' '):
                    if last_word != '':
                        word_count += 1
                    log_prob, _ = log_prob + self.ext_scoring_func(last_word, state, word_count, log=True)
                beam_result.append((log_prob, seq.lstrip()))
            else:
                beam_result.append((float("-INF"), ''))

        ## output top beam_size decoding results
        beam_result = sorted(beam_result, key=lambda asd: asd[0], reverse=True)
        return beam_result[0][1]

    def _get_pruned_log_probs(self, logit, cutoff_prob, cutoff_top_n):
        log_prob_idx = list(enumerate(logit))
        cutoff_len = len(log_prob_idx)
        #If pruning is enabled
        log_prob_idx = sorted(log_prob_idx, key=lambda asd: asd[1], reverse=True)
        if cutoff_prob < 1.0 or cutoff_top_n < cutoff_len:
            if cutoff_prob < 1.0:
                cum_prob = 0.0
                cutoff_len = 0
                for _, log_prob in log_prob_idx:
                    cum_prob += np.exp(log_prob)
                    cutoff_len += 1
                    if cum_prob >= cutoff_prob or cutoff_len >= cutoff_top_n:
                        break
            else:
                cutoff_len = cutoff_top_n
        log_prob_idx = log_prob_idx[0:cutoff_len]      
        return log_prob_idx


        

