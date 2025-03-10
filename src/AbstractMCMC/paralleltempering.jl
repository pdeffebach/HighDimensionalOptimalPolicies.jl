MCMCTempering.getparams_and_logprob(t::PolicyTransition) = t.params, t.obj
function MCMCTempering.setparams_and_logprob!!(t::PolicyTransition, params, obj)
    Policytransition(params, obj, t.accepted)
end