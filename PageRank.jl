using LinearAlgebra
using Random
using Plots
using Statistics

include("random-starting.jl")
include("utils.jl")

Random.seed!(1)
gr(size=(1800,1600))

function MakeStochasticMtrx(n)
A = round.(rand(n,n))
A = LowerTriangular(A)

A = A + A'
A[A .== 2] .= 1

P = NormalizeMtrx(A)

return P
end

function NormalizeMtrx(A)
    col_sums = sum(A,dims=1)
    normedA = A ./ col_sums

    return normedA 
end

n = 1000
P = MakeStochasticMtrx(n)
b = randn(n)
h = 0.9999
q = 200
k = 20

rs_error = RandomStarting(P,b,h,q,k)
PlotVScode(rs_error,"PageRank Random starting","q","error")
