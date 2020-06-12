from numpy import zeros, array
from math import sqrt, log


class JSD():
    def __init__(self):
        self.p = p
        self.q = q
        
    def KL_divergence(self):
        """ Compute KL divergence of two vectors, K(p || q)."""
        return sum(self.p[x] * log((self.p[x]) / (self.q[x])) 
               for x in range(len(self.p)) if self.p[x] != 0.0 or self.p[x] != 0)

    def Jensen_Shannon_divergence(self):
        """ Returns the Jensen-Shannon divergence. """
        JSD = 0.0
        weight = 0.5
        average = zeros(len(self.p)) #Average
        for x in range(len(self.p)):
            average[x] = weight * self.p[x] + (1 - weight) * self.q[x]
            JSD = (weight * KL_divergence(array(self.p), average)) + ((1 - weight) * KL_divergence(array(self.q), average))
        return 1-(JSD/sqrt(2 * log(2)))
