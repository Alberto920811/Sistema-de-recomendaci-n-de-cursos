from numpy import zeros, array
from math import sqrt, log


class JSD():
    def __init__(self):
        self.log2 = log(2)


    def KL_divergence(self, p, q):
        """ Compute KL divergence of two vectors, K(p || q)."""
        return sum(p[x] * log((p[x]) / (q[x])) for x in range(len(p)) if p[x] != 0.0 or p[x] != 0)

    def Jensen_Shannon_divergence(self, p, q):
        """ Returns the Jensen-Shannon divergence. """
        self.JSD = 0.0
        weight = 0.5
        average = zeros(len(p)) #Average
        for x in range(len(p)):
            average[x] = weight * p[x] + (1 - weight) * q[x]
            self.JSD = (weight * self.KL_divergence(array(p), average)) + ((1 - weight) * self.KL_divergence(array(q), average))
        return 1-(self.JSD/sqrt(2 * self.log2))

if __name__ == '__main__':
    J = JSD()
    p = [1.0/10, 1.0/10, 0]
    q = [0, 1.0/10, 9.0/10]
    print(J.Jensen_Shannon_divergence(p, q))