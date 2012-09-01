"""
Provides a class implementing naive Bayesian classification of text content.

Eliot Ball <eb@eliotball.com>
"""


class Classifier:
    """
    Object for performing naive bayesian classification.
    """
    def __init__(self, outcomes):
        "Takes a list of outcomes as strings, constructs classifier."
        # Prepare outcome object list
        self.outcomes = { }
        for outcome in outcomes:
            self.outcomes[outcome] = Outcome()

    def add_training_example(self, outcome, tokens):
        "Takes an outcome and a list of tokens and updates the outcome object."
        self.outcomes[outcome].add_training_example(tokens)
    
    def probability_outcome(self, outcome):
        "Givens P(O) for outcome O."
        return float(self.outcomes[outcome].count) / sum(
            [outcome_obj.count for outcome_name, outcome_obj in self.outcomes.iteritems()])

    def probability_tokens(self, tokens):
        "Gives P(T1 & T2 & ... & Tn) for tokens Ti."
        return sum([outcome_obj.probability_tokens_given_outcome(tokens) 
                    for outcome_name, outcome_obj in self.outcomes.iteritems()])
    
    def probability_outcome_given_token(self, token, outcome):
        "Gives P(O | T) for outcome O and token T."
        # Find the total probability of the token and check it's not zero
        token_probability = (
            sum([self.outcomes[other_outcome].probability_token_given_outcome(token) *
                 self.probability_outcome(other_outcome)
                 for other_outcome in self.outcomes.keys()]))
        if token_probability <= 0.0000001:
            return 0.0
        # Bayes formula
        return (
            self.outcomes[outcome].probability_token_given_outcome(token) *
            self.probability_outcome(outcome) / token_probability)

    def probability_outcome_given_tokens(self, tokens, outcome):
        "Gives P(O | T1 & T2 & ... & Tn) for outcome O and tokens Ti."
        # For each token, calculate the probability of the outcome
        probabilities = [self.probability_outcome_given_token(token, outcome)
                         for token in tokens]
        # Combine the probabilities
        all_true = 1.0
        none_true = 1.0
        for probability in probabilities:
            all_true *= probability
            none_true *= 1.0 - probability
        # Watch out for the classifier failing due to extreme values
        if all_true + none_true <= 0.0000001:
            raise Exception("Divide by zero - consider using more training data.")
        return all_true / (all_true + none_true)

    def classify_tokens(self, tokens):
        "Finds the probability of each outcome given a set of tokens."
        outcome_probabilities = dict(
            [(outcome, self.probability_outcome_given_tokens(tokens, outcome))
             for outcome in self.outcomes.keys()])
        return outcome_probabilities

    def most_likely_outcome(self, tokens):
        "Find the most likely outcome given a set of tokens."
        return max(self.outcomes.keys(), 
            key=lambda outcome: self.probability_outcome_given_tokens(tokens, outcome))


class Outcome:
    """
    Object for representing an outcome and the tokens present when that
    outcome occurs.
    """
    def __init__(self):
        "Constructs empty outcome object."
        self.token_counts = { }
        self.count = 0

    def add_training_example(self, tokens):
        "Takes a token and increments the count for that token."
        # Increment the outcome count
        self.count += 1
        # Increment the count of every token
        for token in tokens:
            if self.token_counts.has_key(token):
                self.token_counts[token] += 1
            else:
                self.token_counts[token] = 1
    
    def probability_token_given_outcome(self, token):
        "Gives P(T | O) for token T and this outcome O."
        if self.token_counts.has_key(token):
            return float(self.token_counts[token]) / self.count
        else:
            return 0.0