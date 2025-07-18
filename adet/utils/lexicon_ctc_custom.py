"""
Custom lexicon-constrained CTC decoding utilities for text recognition.
This module provides functionality to constrain CTC decoding to a predefined vocabulary
with the option to abandon/discard words that don't meet the similarity threshold.
"""

import torch
import numpy as np
from difflib import SequenceMatcher
import editdistance


class LexiconConstrainedCTC:
    """
    Lexicon-constrained CTC decoder that constrains outputs to a predefined vocabulary.
    Words below the similarity threshold are abandoned/discarded.
    """
    
    def __init__(self, vocabulary, voc_size=37, similarity_threshold=0.6):
        """
        Initialize the lexicon-constrained CTC decoder.
        
        Args:
            vocabulary (list): List of allowed words/strings
            voc_size (int): Vocabulary size for CTC (37 or 96)
            similarity_threshold (float): Minimum similarity threshold for lexicon matching
        """
        self.vocabulary = [word.upper() for word in vocabulary]  # Normalize to uppercase
        self.voc_size = voc_size
        self.similarity_threshold = similarity_threshold
        
        # Define character mappings based on vocabulary size
        if self.voc_size == 37:
            self.CTLABELS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
                           'q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
        elif self.voc_size == 96:
            self.CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/',
                           '0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@',
                           'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V',
                           'W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l',
                           'm','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        else:
            raise NotImplementedError(f"Vocabulary size {voc_size} not supported")
            
    def standard_ctc_decode(self, rec):
        """
        Standard CTC decoding without lexicon constraint.
        Matches the original visualizer logic exactly.
        """
        if self.voc_size == 37:
            last_char = -1
            s = ''
            for c in rec:
                c = int(c)
                if c < self.voc_size - 1:  # c < 36 for 37-voc
                    if last_char != c:
                        s += self.CTLABELS[c]
                        last_char = c
                else:
                    last_char = -1  # Reset for blank tokens
        elif self.voc_size == 96:
            last_char = -1
            s = ''
            for c in rec:
                c = int(c)
                if c < self.voc_size - 1:  # c < 95 for 96-voc
                    if last_char != c:
                        s += self.CTLABELS[c]
                        last_char = c
                else:
                    last_char = -1  # Reset for blank tokens
        else:
            raise NotImplementedError
        
        return s.upper()  # Normalize to uppercase
    
    def compute_similarity(self, text1, text2):
        """
        Compute similarity between two texts using multiple metrics.
        """
        # Normalize texts
        text1, text2 = text1.upper(), text2.upper()
        
        # Exact match gets highest score
        if text1 == text2:
            return 1.0
            
        # Character-level similarity using SequenceMatcher
        char_similarity = SequenceMatcher(None, text1, text2).ratio()
        
        # Edit distance based similarity
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            edit_similarity = 1.0
        else:
            edit_distance = editdistance.eval(text1, text2)
            edit_similarity = 1.0 - (edit_distance / max_len)
        
        # Weighted combination (give more weight to character similarity)
        combined_similarity = 0.7 * char_similarity + 0.3 * edit_similarity
        
        return combined_similarity
    
    def find_best_match(self, decoded_text):
        """
        Find the best matching word from vocabulary.
        Returns empty string if no word meets the similarity threshold.
        """
        if not self.vocabulary:
            return decoded_text
    
        best_match = decoded_text
        best_score = 0.0
        
        for vocab_word in self.vocabulary:
            similarity = self.compute_similarity(decoded_text, vocab_word)
            if similarity > best_score:
                best_score = similarity
                best_match = vocab_word
                
        # Only use lexicon match if similarity is above threshold
        if best_score >= self.similarity_threshold:
            return best_match
        else:
            return ""  # Abandon/discard words below threshold
    
    def lexicon_constrained_decode(self, rec):
        """
        Perform lexicon-constrained CTC decoding.
        """
        # First, do standard CTC decoding
        decoded_text = self.standard_ctc_decode(rec)
        
        # If vocabulary is empty, return standard decoding
        if not self.vocabulary:
            return decoded_text
            
        # Find best match in vocabulary
        constrained_text = self.find_best_match(decoded_text)
        
        return constrained_text
    
    def batch_lexicon_decode(self, recs):
        """
        Perform lexicon-constrained decoding on a batch of recognition results.
        """
        results = []
        for rec in recs:
            decoded = self.lexicon_constrained_decode(rec)
            results.append(decoded)
        return results

