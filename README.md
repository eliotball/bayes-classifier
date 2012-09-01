bayes-classifier
================

Introduction
------------

Naive Bayesian classification is a method for determining which set of inputs a particular input is closest to, given a number of sets of 'training' input. For example, a common use of naive Bayes classifiers is in spam filtering, where there is a large set of 'ham messages' (wanted mail), and a large set of 'spam messages' (unwanted mail) and the classifier decides to whether an incoming message is ham or spam.

bayesclassifier.py is a Python implementation of a naive Bayes classifier.

Usage
-----

You must import the Bayes classifier module at the start of the program, and then instantiate a Classifier object, giving a list of possible outcomes.

    import bayesclassifier

    classifier = bayesclassifier.Classifier(["outcome1", "outcome2", "outcome3"])

You should then train your classifier on some sample data for each outcome. Give some tokens that appear in inputs that should have the outcome you specify. The more tokens you add, the better. You can repeatedly call this method to add more training data for each outcome.

    classifier.add_training_example("outcome1", ["a", "list", "of", "words", "or", "other", "tokens"])

After your classifier is trained, you can pass a set of tokens to the classifier, and it will compute which outcome is most likely.

    classifier.most_likely_outcome(["another", "list", "of", "tokens"])

Alternatively, you can obtain a dictionary with all of the possible outcomes, and the probabilities according to the classifier.

    classifier.classify_tokens(["another", "list", "of", "tokens"])
