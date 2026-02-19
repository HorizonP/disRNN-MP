using DataFrames
using HypothesisTests
import Random


""" Data structure for MP task
including algorithm parameters, trial history, and algorithm internal variables for **one** session

"""
struct MatchingPenneisTask
    maxdepth::Integer
    alpha::Real

    "available choice options on each trial"
    choices::Tuple

    history::DataFrame

    """each row represent globally counted bias after one unique histseq

    - histseq: the history choice/outcome sequences preceding a trial
    - detected_on: trial numbers (1-based) whose histseq has triggered the bias detection (statistical significant in the past) (without observation of the monkey's choice on that trial)
    - signif_tris: trial numbers (1-based) which caused a significant bias after the histseq (after observing monkey's choice)
    - matched_on: trial numbers (1-based) whose histseq has matched
    """
    biasCount::DataFrame 
    
    trialBias::DataFrame
    function MatchingPenneisTask(maxdepth::Integer = 4, alpha::Real = 0.05; choices::Tuple = (0,1))
        @assert maxdepth >= 0 "maxdepth has to be non-negative"
        @assert 0 < alpha < 1 "alpha has to be in the range of (0,1)"
        @assert length(choices) == 2 "only 2-armed bandit task is supported"
        return new(maxdepth, alpha, choices, 
            DataFrame(:choice => Integer[], :reward => Integer[], :com_pR => Real[]), # column definition of history
            DataFrame( # column definition of biasCount, this 1st row is for algorithm 0
                :histseq => String[""], 
                :rightCh => Int[0], 
                :total => Int[0], 
                :p_val => Real[NaN], 
                :alg_type => Int[0],
                :depth => Int[0],
                :detected_on => Array{Int64,1}[[]], 
                :signif_tris => Array{Int64,1}[[]],
                :matched_on => Array{Int64,1}[[]]
            ), 
            DataFrame(:detected => Integer[], :depth => Integer[], :magnitude => Real[], :which_alg => Integer[])
        ) 
    end
end

make_MPTs(maxdepth::Integer = 4, alpha::Real = 0.05; choices::Tuple = (0,1), n::Integer = 1) = [MatchingPenneisTask(maxdepth, alpha; choices = choices) for _ in 1:n]

"""
histseq is a string represent recent trial event history from distant (left) to most recent (right).

`depth` determines the number of trials to be used
if `ifOutcome` is false, each trial will be represented by its choice only, such as "011" 
if `ifOutcome` is true, each trial will be represented by choice and outcome, such as "0+1-0-"

if `pos` is not given, generate the history sequence string for last trial in hist
if `pos` is given, generate it at that trial (include that trial in the histseq string)
"""
function histseq_str(hist, depth, ifOutcome; pos::Union{Nothing, Integer} = nothing)
    # [end-depth+1:end, :reward]
    # [end-depth+1:end, :choice]
    if depth<=0
        return ""
    end

    if isnothing(pos)
        pos = nrow(hist)
    end

    chStrVec = string.(hist[pos-depth+1:pos, :choice])
    if ifOutcome
        histseqVec = permutedims([chStrVec ["-", "+"][hist[pos-depth+1:pos, :reward] .+ 1]])[:]
    else
        histseqVec = chStrVec
    end
    return join(histseqVec)
end
# hist = DataFrame(:choice => [1,1,0,1,0,1], :reward => [1,1,1,0,0,1])
# histseq_str(hist, 3, true; pos = 5)
# histseq_str(hist, 3, true)


"""
find the row index of biasCount df with the specific histseq
if it does not exist, create a empty record
"""
function lookup_biasCount!(bc::DataFrame, histseq, depth, ifOutcome)
    # I may maintain a sorted biasCount dataframe to improve the performance here
    # example: sortedStructInsert!(v::Vector, x) = (splice!(v, searchsorted(v,x,by= v->v.date, rev=true), [x]); v)
    I_bc = findfirst(bc.histseq .== histseq)
    if isnothing(I_bc)
        push!(bc, (histseq = histseq, rightCh = 0, total = 0, p_val = NaN, alg_type = (ifOutcome + 1), depth = depth, detected_on = Int[], signif_tris = Int[], matched_on = Int[]))
        I_bc = nrow(bc)
    end
    return I_bc
end


""" This function will (1) examine past history to detect bias, (2) update `biasCount` to assist detection in the future trials, and (3) return a set of detected biases

This function modifies `mpt.biasCount` (but not `mpt.history`)

The `history` argument allows limiting the information computer algorithm can see

Fields of mpt that will be accessed:
- mpt.biasCount
- mpt.maxdepth
- mpt.alpha
- mpt.choices[2]
"""
function detect_and_update_biases!(mpt::MatchingPenneisTask, history::AbstractDataFrame, choice::Integer)

    sgnfctBias = [] # to store all found significant bias by all algorithms
    
    bc = mpt.biasCount # alias
    for dep = 0:mpt.maxdepth, ifOut = [false, true] # loop over algorithms and depths
        # alg-0: depth 0, no outcome
        if (ifOut, dep) != (true, 0) && nrow(history) >= dep

            # ======== lookup potential bias for current trial at varies depth & alg

            histseq = histseq_str(history, dep, ifOut)

            I_bc = lookup_biasCount!(bc, histseq, dep, ifOut)

            curr_tri = nrow(history) + 1
            
            if bc[I_bc, :total] > 0
                p_val = bc[I_bc, :p_val]
                push!(bc[I_bc, :matched_on], curr_tri)
                if p_val < mpt.alpha
                    push!(sgnfctBias, (freq = bc[I_bc, :rightCh] / bc[I_bc, :total], p_val = p_val, ifInclOut = ifOut, depth = dep))
                    push!(bc[I_bc, :detected_on], curr_tri) # this should match trialBias information
                end
            end

            # ======== update mpt.biasCount (note that the computer does not access this information when looking at biases)

            bc[I_bc, :rightCh] += choice == mpt.choices[2]
            bc[I_bc, :total] += 1 
            bc[I_bc, :p_val] = pvalue(BinomialTest(bc[I_bc, :rightCh], bc[I_bc, :total], 0.5); tail = :both)
            if bc[I_bc, :p_val] < mpt.alpha
                push!(bc[I_bc, :signif_tris], curr_tri) # this variable record trials where the bias has become significant after observing current trial choice
            end
        end
    end

    return sgnfctBias

end


"""select the bias to punish

Bias that is most deviated from 0.5 will be used to calculate computer p(R) for punishing

Returns computer p(R), selected bias information
"""
function select_biases(sgnfctBias)
    # examine found biases and determine computer's p(R)
    if length(sgnfctBias) == 0 # no bias detected
        p_comp_chx = 0.5
        bias_info = (detected = 0, depth = -1, magnitude = 0, which_alg = 0)
    else
        biases = DataFrame(sgnfctBias) # N x 4 matrix

        # ========== below are bias selection algorithm =============

        devs_sign = biases[:, :freq] .- 0.5
        devs = abs.(devs_sign)
        I = argmax(devs) # find the (index of *first*) biased freq which is most deviated from 0.5
        p_comp_chx = 1 - biases[I, :freq] # computer is more likely to choose the opposite
        
        bias_info = (
            detected = if biases[I, :freq] > 0.5 2 else 1 end,
            depth = biases[I, :depth],
            magnitude = biases[I, :freq] - 0.5,
            which_alg = if biases[I, :ifInclOut] 2 else 1 end
        )
    end
    return p_comp_chx, bias_info
end

"""
This function takes task variables and agent choice on current trial,
it first determine computer's choice without using agent's current choice information
Then, it will update task variables (tracked agent biases) with agent's choice. 
Finally, it output agent's outcome determined by computer and agent's choice 

## Implementation details

look into a cached ledger of biases to determine computer's strategy
"""
function run_step(mpt::MatchingPenneisTask, choice::Integer)
    @assert choice ∈ mpt.choices "the agent has to choose among $(mpt.choices)"


    # ============== matching pennies algorithm ==============
    sgnfctBias = detect_and_update_biases!(mpt, mpt.history, choice)
    p_comp_chx, bias_info = select_biases(sgnfctBias)

    # computer make choice
    comp_choice = if rand() < p_comp_chx mpt.choices[2] else mpt.choices[1] end

    # determine reward and update history
    rew = comp_choice == choice
    push!(mpt.history, (choice = choice, reward = rew, com_pR = p_comp_chx))
    push!(mpt.trialBias, bias_info)

    return rew
end

"""
generate histseq string from trialBias info
"""
function add_histseq_col(mpt::MatchingPenneisTask)
    histseqs = String[]
    for i = 1:nrow(mpt.trialBias)
        if mpt.trialBias[i, :detected] > 0
            ifOut = if mpt.trialBias[i, :which_alg] == 2 true else false end
            push!(histseqs, histseq_str(mpt.history, mpt.trialBias[i, :depth], ifOut; pos = i-1))
        else
            push!(histseqs, "")
        end
    end

    mpt.trialBias[!,:histseq] = histseqs

end


function cal_MP_algs(mpt::MatchingPenneisTask)

    # ============== matching pennies algorithm ==============
    bc = mpt.biasCount # alias

    # update alg 0 variable from 1st trial
    I_bc = lookup_biasCount!(bc, "", 0, false)
    bc[I_bc, :rightCh] += mpt.history[1,:choice] == mpt.choices[2]
    bc[I_bc, :total] += 1 
    p_comp_chx = 0.5
    bias_info = (detected = 0, depth = -1, magnitude = 0, which_alg = 0)
    mpt.history[1,:com_pR] = p_comp_chx
    push!(bc[I_bc, :matched_on], 1)
    push!(mpt.trialBias, bias_info)

    for tri = 2:nrow(mpt.history)
        # tri is current trial, for which the following code will calculate algorithm's output on this trial (without knowing current trial monkey choice)

        sgnfctBias = detect_and_update_biases!(
            mpt, 
            view(mpt.history, 1:tri-1, :),  # only has view for history up to last trial
            mpt.history[tri, :choice])

        p_comp_chx, bias_info = select_biases(sgnfctBias)

        mpt.history[tri,:com_pR] = p_comp_chx
        push!(mpt.trialBias, bias_info)
    end

end


# import PythonCall
# function batch_run_step(mpts, choices)

#     rews = zeros(Integer, length(mpts))
#     PythonCall.GIL.@unlock Threads.@threads for i=range(1, length(mpts))
#         rews[i] = run_step(mpts[i], Integer(choices[i]))
#     end
    
#     return rews
# end


# # initialize
# mpt = MatchingPenneisTask()
# run_step(mpt, rand([0, 1]))

# # for testing
# function runMP(n = 1000)
#     mpt = MatchingPenneisTask()

#     for i = 1:n
#         run_step(mpt, rand([0, 1]))
#     end
#     return mpt
# end

# mpt = runMP(15000)

# # mpt.biasCount

# mpt.trialBias
# add_histseq_col(mpt)
