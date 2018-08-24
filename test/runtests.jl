using BayesInference
using Test
using Random

# write your own tests here

x = collect(1:3)*1.0
@test logsumexp(x) ≈ 3.40760596

x = reshape(collect(1:24)*1.0, (2,3,4))
@test sum(abs.(logsumexp(x, 2) .- reshape([5.14293 6.14293 11.1429 12.1429 17.1429 18.1429 23.1429 24.1429], (2, 1, 4)))) < 0.0002

Random.seed!(1)
@test sum(abs.(rDirichlet(ones(2),true) .- [-0.742854, -0.645794])) < 0.00001

@test all(embed(collect(1:5), 2) .≈ [i-j for i=3:5, j=0:2] * 1.0)

Random.seed!(1)
xx = randn(10, 5)
Sig = xx'xx
mu = collect(1:5)*1.0
x_indx = 1:3
y_indx = 4:5
μx = mu[x_indx]
μy = mu[y_indx]
Σxx = Sig[x_indx, x_indx]
Σyy = Sig[y_indx, y_indx]
Σxy = Sig[x_indx, y_indx]
x = [0.3, 2.1, 3.4]
cn = condNorm(μx, μy, Σxx, Σxy, Σyy, x)
cn2 = MvNormal([4.05777, 5.20685], [10.7888 1.43842; 1.43842 5.80639])
@test sum(abs.(cn.μ .- cn2.μ)) < 0.0001
@test sum(abs.(Matrix(cn.Σ) .- Matrix(cn2.Σ))) < 0.01

@test ldnorm(0.5, 0.0, 1.0) ≈ -1.0439385332046727

x = [1.0, 2.0]
@test lmvbeta(x) ≈ -0.6931471805599453




ȳ = 1.0
n = 2
σ2 = 1.0
μ0 = 0.0
v0 = 100.0
post = post_norm_mean(ȳ, n, σ2, μ0, v0)
@test post.μ ≈ 0.9950248756218907
@test post.σ ≈ 0.7053456158585983

ss = 100.0
n = 10
a0 = 1.0
b0 = 1.0
post = post_norm_var(ss, n, a0, b0)
@test mean(post) ≈ 10.2 && var(post) ≈ 26.01


Random.seed!(1)
n = 10
k = 2
X = randn(n,k)
y = randn(n)
σ2 = 1.0
τ2 = 2.0
β0 = 0.5
Random.seed!(1)
@test rpost_normlm_beta1(y, X, σ2, τ2, β0) ≈ [0.053687502972789736, 0.39201537149695614]

Random.seed!(1)
@test rpost_normlm_beta1(y[1], X[1,:], σ2, τ2, β0) ≈ [-0.4528229833594598, -0.9704987293677494]

Random.seed!(1)
@test rpost_normlm_beta1(y, X[:,1], σ2, τ2, β0)[1] ≈ 0.11759823040187813

Random.seed!(1)
@test rpost_normlm_beta1(y[1], X[1,1], σ2, τ2, β0)[1] ≈ -0.7390497060177426






@test logSDMweights([1.0, 1.5], 2.0) ≈ [-1.056052674249314, -0.4274440148269396]

@test logSDMmarginal([0,2], [0.1, 1.0], 2.0) ≈ -0.10189829456439084

Random.seed!(1)
@test rSparseDirMix([1.0, 1.5], 2.0, false) ≈ [0.8646629261169209, 0.13533707388307914]

Random.seed!(1)
post = rpost_sparseStickBreak([0,2], 0.65, 10.0, 0.80, 5.0, false)
@test post[1] ≈ [0.07781921103773391, 0.9221807889622661]
@test post[2][1] ≈ 0.07781921103773391
@test post[3][1] == 1

Random.seed!(1)
post = rpost_sparseStickBreak([0,2], 0.65, 10.0, 0.80, 5.0, 1.0, 1.0, false)
@test post[1] ≈ [0.07781921103773391, 0.9221807889622661]
@test post[2][1] ≈ 0.07781921103773391
@test post[3][1] == 1
@test post[4] ≈ 0.5113871699204988

Random.seed!(1)
post = rpost_sparseStickBreak([0,2], 0.65, 10.0, 0.80, 5.0, 1.0, 1.0, 5.0, 1.0, false)
@test post[1] ≈ [0.07781921103773391, 0.9221807889622661]
@test post[2][1] ≈ 0.07781921103773391
@test post[3][1] == 1
@test post[4] ≈ 0.826909596917828
@test post[5] ≈ 0.04808366016426602

@test logSBMmarginal([0,2], 0.65, 10.0, 0.8, 5.0) ≈ -0.5709295478356963


## test for Bitbucket
