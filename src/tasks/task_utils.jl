function ratdata_tasks(task::Union{AbstractString,Nothing}=nothing)
    task_types = subtypes(RatData)
    task_strings = string.(task_types)
    tasks = Dict([ts=>tt for (ts,tt) in zip(task_strings,task_types)])
    if !isnothing(task)
        return tasks[task]
    else
        return tasks
    end
end


"""
    simulate(sim_options::S,model_options::T,agent_options::AgentOptions;init_hypers::Bool=true,return_z=false,seed=nothing) where {S <: SimOptions, T <: ModelOptions}

Simulates task data according to `sim_options`, generated by model specified by `model_options` and `agent_options`.

Optional arguments:
- `init_hypers`: if `true`, initialize model hyperparameters; otherwise, use hyperparameters specified in `model_options`
- `return_z`: default = `false`. if `true`, return generated latent state sequence `z` in addition to task data
- `seed`: seed for random number generator (default = `nothing`)
"""
function simulate(sim_options::S,model_options::T,agent_options::AgentOptions;init_model::Bool=true,return_z=false,seed=nothing,agents_prior=nothing) where {S <: SimOptions, T <: ModelOptions}
    
    new_sess,ntrials,sim_options = simulate(sim_options,seed)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    if init_model
        agents = simulate(agent_options)
        if !isnothing(agents_prior)
            model = simulate(model_options,agents_prior;ntrials=ntrials,new_sess=new_sess)
        else
            model = simulate(model_options,agents;ntrials=ntrials,new_sess=new_sess)
        end
    else
        agents = agent_options.agents
        model = initialize(model_options,agents;ntrials=ntrials,new_sess=new_sess)
    end

    if return_z
        data,z = simulate(model,agents,sim_options,new_sess;return_z=return_z,seed=seed)
    return data, model, agents, z
    else
        data = simulate(model,agents,sim_options,new_sess;seed=seed)
        return data, model, agents
    end
end

function simulate(sim_options::S,model::T,agents::Array{A};return_z=false,seed=nothing) where {S <: SimOptions, T <: MixtureAgentsModel, A <: Agent}
    new_sess,_,sim_options = simulate(sim_options,seed)
    if return_z
        data,z = simulate(model,agents,sim_options,new_sess;return_z=return_z,seed=seed)
        return data,z
    else
        data = simulate(model,agents,sim_options,new_sess;seed=seed)
        return data
    end
end

function simulate(sim_options::S,seed=nothing) where S <: SimOptions
    @unpack nsess, ntrials, mean_ntrials, std_ntrials = sim_options
    if !isnothing(seed)
        Random.seed!(seed)
    end
    # session breaks
    if mean_ntrials > 0
        ntrials_per_sess = Int.(round.(rand(Normal(mean_ntrials,std_ntrials), nsess)))
        ntrials = sum(ntrials_per_sess)
        sim_options = Parameters.reconstruct(sim_options, ntrials=ntrials)
    else
        ntrials_per_sess = repeat([Int.(round(ntrials/nsess))],nsess)
    end
    new_sess = falses(ntrials)
    sess_start = 1
    for i = 1:nsess
        new_sess[sess_start] = true
        sess_start = sess_start + ntrials_per_sess[i]
    end

    return new_sess,ntrials,sim_options

end

##################################################################################
########--  generally helpful functions for task-specific simulation  --##########
##################################################################################

"""
    sim_choice_prob(model,x,trial,pz,new_sess)

Computes choice probability and latentent state probability for next trial
For MoA-HMM
"""
function sim_choice_prob(model::ModelHMM,x,trial,pz,new_sess)
    @unpack β,π,A = model
    if new_sess[trial]
        pz = copy(π)
    end
    z = searchsortedfirst.(Ref(cumsum(pz)),rand())
    if (size(x,1)==length(β)) & (size(β,2) > 1)
        py = logistic(βT_x(β[:,z],reshape(x,(size(β)...,:))[:,z,trial])[1])
    else
        py = logistic(βT_x(β[:,z],x[:,trial])[1])
    end
    pz = A[z,:]
    return py, pz, z
end

"""
    sim_choice_prob(model,x,trial,pz,new_sess)

Computes choice probability and latentent state probability for next trial
For MoA-β model (latent state probability defaults to 1)
"""
function sim_choice_prob(model::ModelDrift,x,trial,pz=1,new_sess=false)
    @unpack β = model
    py = logistic(βT_x(β[:,trial],x[:,trial])[1])
    return py, pz, 0
end


"""
    get_sess_inds(sessions)

Function to get session index ranges from boolean vector `new_sess`
"""
function get_sess_inds(new_sess)
    inds = vcat(findall(new_sess),length(new_sess)+1)
    trials = 1:length(new_sess)
    return getindex.(Ref(trials), (:).(inds[1:end-1],inds[2:end] .- 1))
end

"""
    use_sessions(sessions, new_sess)

Returns a boolean vector of size `new_sess` with `true` values corresponding to trials within `sessions`
"""
function use_sessions(sessions::T1,new_sess::T2,sess_inds::Any) where {T1 <: Any, T2}
    sess_use = falses(size(new_sess))
    for sess_i in sess_inds[sessions]
        if length(sess_i) > 1
            sess_use[sess_i] .= true
        else
            sess_use[sess_i] = true
        end
    end
    return sess_use
end

"""
    trim_sessions!(field,sess_use)

Trim entries of `field` to only include those specified by `sess_use`
"""
function trim_sessions!(field::AbstractVector{T},sess_use) where T
    if VERSION < v"1.7"
        not_use = not_sessions(sess_use)
        if T <: Bool
            deleteat!(field,findall(not_use))
        else
            deleteat!(field,not_use)
        end

    else
        keepat!(field,sess_use)
    end
end

