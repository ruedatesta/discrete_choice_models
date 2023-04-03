using Plots, Distributions, Random, Optim, BenchmarkTools, Plots

Random.seed!(3)
β=0.5
μ=2.0;
n=1000;
x=rand(Uniform(1, 10),n)

dist=Gumbel();
ϵw=rand(dist,n);
ϵn=rand(dist,n);

uw=β*x+ϵw;
un=[μ].+ϵn;

decision=uw.>un

mean(decision)

function prob(β,μ,x)
    pr=1/(1+exp(μ-β*x))
end

"""
    logL_fn(θ)

Compute the log likelihood function, given
values of `β` and `μ`.

# Arguments

- `β::Float64`: education premium.
- `μ::Float64`: deterministic component of leisure.

"""
function logL_fn(θ) #LogL function
    β=θ[1]
    μ=θ[2]
    logL=0
    pr=prob.(β,μ,x)
    logL=(log.(pr).*decision).+(log.([1].-pr)).*([1].-decision)
    return -(sum(logL))
end

θguess=[0.4,1.0];

res=optimize(logL_fn, θguess)
θstar=Optim.minimizer(res)
## Here it ends the function.