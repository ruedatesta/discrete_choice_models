using Plots, Distributions, Random, Optim, BenchmarkTools


Random.seed!(3)
β=0.5
μ=2.0;
n=10000
x=rand(Uniform(1, 10),n)

dist=Normal();
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

z=Normal(0,sqrt(2))
function prob(β,μ)
    pr=cdf(z,β*x.-[μ])
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
res=optimize(logL_fn, θguess)
θstar=Optim.minimizer(res)