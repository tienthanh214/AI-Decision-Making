## MULTI-CAREGIVER CRYING BABY
# Partially Observable Markov Games

# ------ IMPORT ----------
import Pkg
if !haskey(Pkg.installed(), "JuMP") 
    Pkg.add("JuMP")
end
if !haskey(Pkg.installed(), "Ipopt")
    Pkg.add("Ipopt")
end

using JuMP, Ipopt, Random

# ------ INIT ----- 

SING = "SING"
CRYING = "CRYING"
QUIET = "QUIET"
FEED = "FEED"
SATED = "SATED"
HUNGRY = "HUNGRY"

p_cry_when_hungry_in_sing = 0.9
p_cry_when_hungry = 0.9
p_cry_when_not_hungry = 0.0
p_become_hungry = 0.5

r_hungry = 10.0
r_sing = 0.5
r_feed = 5.0

struct POMG
    Ī³  # discount factor
    ā  # agents
    š®  # state space
    š  # joint action space 
    šŖ  # joint observation space
    T  # transition function
    O  # joint observation function
    R  # joint reward function

    function POMG(discount, agents, states, jointAction, jointObservation, transitionFunc, jointObservationFunc, jointRewardFunc)
        new(discount, agents, states, jointAction, jointObservation, transitionFunc, jointObservationFunc, jointRewardFunc)
    end
end

struct ConditionalPlan
    a   # action to take at root
    subplans    # dictionary mapping observations to subplans 
end

struct SimpleGame
    Ī³  # discount factor
    ā  # agents
    š  # joint action space
    R  # joint reward function
end

struct NashEquilibrium end

struct SimpleGamePolicy
    p # dictionary mapping actions to probabilities
    
    # Returns a random policy
    function SimpleGamePolicy(p::Base.Generator)
        return SimpleGamePolicy(Dict(p))
    end

    # Return policy from dict
    function SimpleGamePolicy(p::Dict)
        vs = collect(values(p))
        vs ./= sum(vs)
        return new(Dict(k => v for (k,v) in zip(keys(p), vs)))
    end

    SimpleGamePolicy(ai) = new(Dict(ai => 1.0))
end

ConditionalPlan(a) = ConditionalPlan(a, Dict())

(Ļ::ConditionalPlan)() = Ļ.a
(Ļ::ConditionalPlan)(o) = Ļ.subplans[o]

# ---------- TRANSITION FUNCTION ----------
function transition(s, a, sā²)
    # Regardless, feeding makes the baby sated.
    if a[1] == "FEED" || a[2] == "FEED" 
        if sā² == "SATED" 
            return 1.0
        else 
            return 0.0
        end
    else
        # If neither caretaker feed, then one of two things happens.
        # First, a baby that is hungry remains hungry 
        if s == "HUNGRY"
            if sā² == "HUNGRY"
                return 1.0
            else 
                return 0.0
            end
        # Otherwise, it becomes hungry with a fixed probability.
        else
            if sā² == "SATED"
                return 1.0 - p_become_hungry
            else
                return p_become_hungry
            end 
        end 
    end
end

# ---------- JOINT OBSERVATION FUNCTION --------- 
function joint_observation(a, sā², o)
    # If at least one caregiver sings, then both observe the result.
    if a[1] == "SING" || a[2] == "SING"
        # If the baby is hungry, then the caregivers both observe crying/silent together.
        if sā² == "HUNGRY"
            if o[1] == "CRYING" && o[2] == "CRYING"
                return p_cry_when_hungry_in_sing
            elseif o[1] == "QUIET" && o[2] == "QUIET"
                return 1.0 - p_cry_when_hungry_in_sing
            else 
                return 0.0
            end
        # Otherwise the baby is sated
        else
            if o[1] == "QUIET" && o[2] == "QUIET"
                return 1.0
            else 
                return 0.0
            end
        end
    # Otherwise the caregivers fed and/or ignored the baby
    else 
        # If the baby is hungry, then thereā²s a probability it cries
        if sā² == "HUNGRY"
            if o[1] == "CRYING" && o[2] == "CRYING"
                return p_cry_when_hungry 
            elseif o[1] == "QUIET" && o[2] == "QUIET"
                return 1.0 - p_cry_when_hungry
            else 
                return 0.0
            end 
        # If the baby is sated, then thereā²s no probability it cries
        else
            if o[1] == "CRYING" && o[2] == "CRYING" 
                return p_cry_when_not_hungry
            elseif o[1] == "QUIET" && o[2] == "QUIET"
                return 1.0 - p_cry_when_not_hungry
            else 
                return 0.0
            end
        end 
    end
end

# -------------- JOINT REWARD FUNCTION -----------------
function joint_reward(s, a) 
    r = [0.0, 0.0]
    
    # Both caregivers do not want the child to be hungry
    if s == "HUNGRY"
        r -= [r_hungry, r_hungry]
    end

    # the first caregiver favors feeding 
    if a[1] == "FEED" 
        r[1] -= r_feed / 2.0 
    elseif a[1] == "SING"
        r[1] -= r_sing
    end

    # the second caregiver favors singing
    if a[2] == "SING"
        r[2] -= r_sing / 2
    elseif a[2] == "FEED"
        r[2] -= r_feed
    end
    
    return r
end

# -------------- EVALUATIONG CONDITIONAL PLANS --------------
#  The lookahead function below is used to calculate the evaluate plan
function lookahead(š«::POMG, U, s, a)
    š®, šŖ, T, O, R, Ī³ = š«.š®, joint(š«.šŖ), š«.T, š«.O, š«.R, š«.Ī³
    uā² = sum(T(s,a,sā²)*sum(O(a,sā²,o)*U(o,sā²) for o in šŖ) for sā² in š®)
    return R(s,a) + Ī³*uā²
end

#  The lookahead function below is used to calculate the utility
function evaluate_plan(š«::POMG, Ļ, s)
    a = Tuple(Ļi() for Ļi in Ļ)
    U(o,sā²) = evaluate_plan(š«, [Ļi(oi) for (Ļi, oi) in zip(Ļ,o)], sā²)
    return isempty(first(Ļ).subplans) ? š«.R(s,a) : lookahead(š«, U, s, a)
end

# used to calculate utility with initial belief b when executing joint policy in POMG š«
function utility(š«::POMG, b, Ļ)
    u = [evaluate_plan(š«, Ļ, s) for s in š«.š®]
    return sum(bs * us for (bs, us) in zip(b, u))
end

#  ----------- NASH EQUILIBRIUM ------------

function expand_conditional_plans(š«, Ī )
    ā, š, šŖ = š«.ā, š«.š, š«.šŖ
    return [[ConditionalPlan(ai, Dict(oi => Ļi for oi in šŖ[i]))
        for Ļi in Ī [i] for ai in š[i]] for i in ā]
end

joint(X) = vec(collect(Iterators.product(X...)))
joint(Ļ, Ļi, i) = [i == j ? Ļi : Ļj for (j, Ļj) in enumerate(Ļ)]

# Returns the format tensor of š«
function tensorform(š«::SimpleGame)
    ā, š, R = š«.ā, š«.š, š«.R
    āā² = eachindex(ā)
    šā² = [eachindex(š[i]) for i in ā]
    Rā² = [R(a) for a in joint(š)]
    return āā², šā², Rā²
end

# Find the Nash Equilibrium
function solve(M::NashEquilibrium, š«::SimpleGame)
    ā, š, R = tensorform(š«)
    model = Model(Ipopt.Optimizer)
    #  declaration
    @variable(model, U[ā])
    # constraint 3
    @variable(model, Ļ[i=ā, š[i]] ā„ 0)
    # objective function
    @NLobjective(model, Min,
        sum(U[i] - sum(prod(Ļ[j,a[j]] for j in ā) * R[y][i]
            for (y,a) in enumerate(joint(š))) for i in ā))
    # constraint 1
    @NLconstraint(model, [i=ā, ai=š[i]],
        U[i] ā„ sum(
            prod(j==i ? (a[j]==ai ? 1.0 : 0.0) : Ļ[j,a[j]] for j in ā)
            * R[y][i] for (y,a) in enumerate(joint(š))))
    # constrain 2
    @constraint(model, [i=ā], sum(Ļ[i,ai] for ai in š[i]) == 1)
    # Model optimization
    optimize!(model)
    Ļiā²(i) = SimpleGamePolicy(š«.š[i][ai] => value(Ļ[i,ai]) for ai in š[i])
    return [Ļiā²(i) for i in ā]
end 

# ------------ DYNAMIC PROGRAMING -------------
struct POMGDynamicProgramming
    b # initial belief
    d # depth of conditional plans
end

# Dynamic programming computes a Nash equilibrium Ļ for a POMG š«, given an initial belief b and horizon depth d. 
function solve(M::POMGDynamicProgramming, š«::POMG)
    ā, š®, š, R, Ī³, b, d = š«.ā, š«.š®, š«.š, š«.R, š«.Ī³, M.b, M.d
    Ī  = [[ConditionalPlan(ai) for ai in š[i]] for i in ā]
    for t in 1:d
        Ī  = expand_conditional_plans(š«, Ī )
        prune_dominated!(Ī , š«)
    end
    š¢ = SimpleGame(Ī³, ā, Ī , Ļ -> utility(š«, b, Ļ))
    Ļ = solve(NashEquilibrium(), š¢)
    return Tuple(argmax(Ļi.p) for Ļi in Ļ)
end

# use to cut branch
function prune_dominated!(Ī , š«::POMG)
    done = false
    while !done
        done = true
        for i in shuffle(š«.ā)
            for Ļi in shuffle(Ī [i])
                if length(Ī [i]) > 1 && is_dominated(š«, Ī , i, Ļi)
                    filter!(Ļiā² -> Ļiā² ā  Ļi, Ī [i])
                    done = false
                    break
                end
            end
        end
    end
end

# used to determine which branch is dominated by another branch
function is_dominated(š«::POMG, Ī , i, Ļi)
    ā, š® = š«.ā, š«.š®
    jointĪ noti = joint([Ī [j] for j in ā if j ā  i])
    Ļ(Ļiā², Ļnoti) = [j==i ? Ļiā² : Ļnoti[j>i ? j-1 : j] for j in ā]
    Ui = Dict((Ļiā², Ļnoti, s) => evaluate_plan(š«, Ļ(Ļiā², Ļnoti), s)[i]
            for Ļiā² in Ī [i], Ļnoti in jointĪ noti, s in š®)
    model = Model(Ipopt.Optimizer)
    @variable(model, Ī“)
    @variable(model, b[jointĪ noti, š®] ā„ 0)
    @objective(model, Max, Ī“)
    @constraint(model, [Ļiā²=Ī [i]],
        sum(b[Ļnoti, s] * (Ui[Ļiā², Ļnoti, s] - Ui[Ļi, Ļnoti, s])
        for Ļnoti in jointĪ noti for s in š®) ā„ Ī“)
    @constraint(model, sum(b) == 1)
    optimize!(model)
    return value(Ī“) ā„ 0
end

multicare = POMG(0.9, 
                [1, 2], 
                ["HUNGRY", "SATED"], 
                [["FEED", "SING", "IGNORE"], ["FEED", "SING", "IGNORE"]], 
                [["CRYING", "QUIET"], ["CRYING", "QUIET"]], 
                transition, 
                joint_observation, 
                joint_reward);

b = [0.5, 0.5];

dyP = POMGDynamicProgramming(b, 1);
result = solve(dyP, multicare);
print(result)