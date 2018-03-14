

## 2.1 Modélisation probabiliste

### 1)

import openturns as ot

L = ot.Uniform(96, 110)
delta_Hw = ot.Uniform(28, 32)
gamma_g = ot.Uniform(3700, 4200)
parameters = ot.LogNormalMuSigma(10**-3, 2*10**-3)
k_er = ot.ParametrizedDistribution(parameters)
tau_c = ot.TruncatedNormal(20, 5, 12, 28)
Rd = ot.Uniform(0.25, 0.35)
Rmax = ot.Uniform(55, 65)

g = 9.81
gamma_w = 1000

### 2)
Z = 2 * L / (gamma_w * delta_Hw)

def draw_Z_pdf():
    Z_mean = Z.getMean()[0]
    Z_sd = Z.getStandardDeviation()[0]
    graph = Z.drawPDF(Z_mean - 3*Z_sd, Z_mean + 3*Z_sd, 100)
    graph.setTitle('Z PDF')
    # the following line causes a crash
    # graph.setLegends([]) 
    # this one works
    graph.setLegendPosition('')
    ot.Show(graph)

### 3) 
# D'après l'énoncé, les variables aléatoires définies à la question 1) 
# sont supposées indépendantes, le vecteur X que l'on doit définir est 
# donc composés de v.a. indépandantes.
# On peut donc laisser le constructeur de ComposedDistribution utiliser 
# une IndependentCopula comme dernier paramètre.
marginals = [Z, gamma_g, k_er, tau_c, Rd, Rmax]
X = ot.ComposedDistribution(marginals)

### 4)
X_loi = X
X = ot.RandomVector(X_loi)

### 5)
print("mean(X) : ", X.getMean())
covs = X.getCovariance()
dim = covs.getDimension()
import math
print("sd(X) : ",  [math.sqrt(covs[(i,i)]) for i in range(dim)])


## 2.2 Création de la variable d'intérêt

### 1)
# Z = 2 * L / (gamma_w * delta_Hw)
# Rmin = 2 * L * tau_c / (gamma_w * delta_Hw)
#      = Z * tau_c
# t_er = 2 * gamma_g * L / (g * gamma_w * k_er * delta_Hw)
#      = Z * gamma_g / (g * k_er)
# delta_tu = t_er log( (Rmax - Rmin) / (Rd - Rmin) )
f = ot.SymbolicFunction(['Z', 'gamma_g', 'k_er', 'tau_c', 'Rd', 'Rmax'],
                        ['Z * gamma_g / (9.81 * k_er) * log( (Rmax - Z * tau_c) / (Rd - Z * tau_c) )'])

### 2)
Y = ot.RandomVector(f, X)

### 3)
n = 10000
samples = Y.getSample(n)
mean = samples.computeMean()
sd = samples.computeStandardDeviation()
print('Mean of {} realizations of Y :'.format(n), mean)
print('Standard deviation of {} realizations of Y :'.format(n), sd)

def show_kernel_smoothing_pdf():
    kernel = ot.KernelSmoothing()
    fittedDist = kernel.build(samples)
    graph = fittedDist.drawPDF()
    ot.Show(graph)


## 2.3 Calcul de probabilité d’événements rares

E1 = ot.Event(Y, ot.Less(), 90 * 60) # E1 : le barrage s'effondre avant 1h30 (durée nécessaire à l'évacuation)
E2 = ot.Event(Y, ot.Greater(), 180 * 60) # E2 : le barrage peut être stabilisé

#### 1)
print("FORM")

solver = ot.SQP()
form = ot.FORM(solver, E1, X.getMean())
c1, c2, c3 =  f.getCallsNumber(), f.getGradientCallsNumber(), f.getHessianCallsNumber()
form.run()
result = form.getResult()
print("Probability of E1 :", result.getEventProbability())
print('Number of calls to f:', f.getCallsNumber() - c1)
print('Number of calls to f.gradient:', f.getGradientCallsNumber() - c2)
print('Number of calls to f.hessian:', f.getHessianCallsNumber() - c3)

solver = ot.SQP()
form = ot.FORM(solver, E2, X.getMean())
c1, c2, c3 =  f.getCallsNumber(), f.getGradientCallsNumber(), f.getHessianCallsNumber()
form.run()
result = form.getResult()
print("Probability of E2 :", result.getEventProbability())
print('Number of calls to f:', f.getCallsNumber() - c1)
print('Number of calls to f.gradient:', f.getGradientCallsNumber() - c2)
print('Number of calls to f.hessian:', f.getHessianCallsNumber() - c3)


print("SORM")

solver = ot.SQP()
sorm = ot.SORM(solver, E1, X.getMean())
c1, c2, c3 =  f.getCallsNumber(), f.getGradientCallsNumber(), f.getHessianCallsNumber()
sorm.run()
result = sorm.getResult()
print("Probability of E1 :", result.getEventProbabilityBreitung())
print('Number of calls to f:', f.getCallsNumber() - c1)
print('Number of calls to f.gradient:', f.getGradientCallsNumber() - c2)
print('Number of calls to f.hessian:', f.getHessianCallsNumber() - c3)

solver = ot.SQP()
sorm = ot.SORM(solver, E2, X.getMean())
c1, c2, c3 =  f.getCallsNumber(), f.getGradientCallsNumber(), f.getHessianCallsNumber()
sorm.run()
result = sorm.getResult()
print("Probability of E2 :", result.getEventProbabilityBreitung())
print('Number of calls to f:', f.getCallsNumber() - c1)
print('Number of calls to f.gradient:', f.getGradientCallsNumber() - c2)
print('Number of calls to f.hessian:', f.getHessianCallsNumber() - c3)


