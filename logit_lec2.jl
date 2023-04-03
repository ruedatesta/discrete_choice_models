using Plots, Distributions, Random, Optim, BenchmarkTools, StatsBase

Random.seed!(3)
β=0.5
μ=2.0;
n=1000
x=rand(Uniform(1, 10),n)

dist=Gumbel();
ϵw=rand(dist,n);
ϵn=rand(dist,n);

uw=β*x+ϵw;
un=ϵn.+μ;

decision=zeros(n)
for i=1:n
    if uw[i]>un[i]
        decision[i]=1
    else
        decision[i]=0
    end
end
mean(decision)

sim=2000;

ϵw_sim=rand(dist,sim)
ϵn_sim=rand(dist,sim)

z_sim=ϵn_sim-ϵw_sim;
z_sim=sort!(z_sim)

z_cdf=ecdf(z_sim)

z_cdf(5)

function prob(β,μ)
    pr=z_cdf(β*x.-[μ])
end


function logL_fn(θ) #LogL function
    β=θ[1]
    μ=θ[2]
    logL=0
    n=1000
    pr=prob(β,μ)
    for id=1:n 
        logL=logL+log(pr[id])*decision[id]+log(1-pr[id])*(1-decision[id])
    end
    return -(logL)
end

θguess=[0.4,1.0]

@btime begin
res=optimize(logL_fn, θguess)
θstar=Optim.minimizer(res)
end