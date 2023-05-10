# Classification using CPC codes of patents

We use the CPC classifications to get a better idea of which algorithms are better
at picking up whether patents are in similar classification classes.

## CPC classification similarity

First we need to determine a measure of how similar two CPC classifications are. Let's
investigate the structure of CPC classification with the use of an example: Y10T74/2173
The CPC classifications are hierarchical in nature, where the characters/numbers on the
at the start denote more general classifications. We can split the classification in
6 separate stages: [Y][1][0][T][74]/[2173], where [Y] is the most general and [2173] is
the most specific.

To compute the similarity we look up to which level (i) the two classifications are the same.
For example, Y10S101/35 is the same for the first 3 places, while C21C5/04 is the same
for 0 places (i.e. not similar at all). The similarity is then computes as 1-a^i, where
a is the similarity exponent. We have put this value as 2/3.

## Patent similarity

Patents generally have multiple CPC classes assigned to them. The idea is to check for each
CPC classification, the most similar classification of the other patent. Let's take an example again:

Patent i:
Y10T74/2173
B41F1/30

Patent j:
B41F1/40
F16C11/02
Y10S101/35

For patent i, Y10T74/2173 is most similar to patent Y10S101/35 in patent j, which gives a similarity of 1-(2/3)^3.
B41F1/30 is most similar to B41F1/40 in patent j, which results in 1-(2/3)^5. We do the same for patent j.
The patent similarity is then simply the average of all these numbers.
