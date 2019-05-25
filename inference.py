import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    # TODO: Compute the forward messages
    #forward messages are a distribution
    #update distribution based on observation and transition model

    print("COMPUTING ALPHA VALUES NOW")
    alpha_z0 = prior_distribution
    for state in prior_distribution:
        likelihood_function = observation_model(state) #p(xi, yi| z0)
        p_obs_given_hidden = likelihood_function[observations[0]] #p(first observation | z0)
        alpha_z0[state] = prior_distribution[state]*p_obs_given_hidden #p(first observation|z0) * p(z0) proportional to p(z0|x0,y0)
    alpha_z0.renormalize()
    forward_messages[0] = alpha_z0


    for i in range(1,num_time_steps):

        alpha_zi_1 = forward_messages[i-1] #just want the shapes to match previous timestep distribution
        xi_yi = observations[i]
        alpha_zi = rover.Distribution()

        for zi in all_possible_hidden_states:
            sum = 0
            likelihood_function = observation_model(zi) #p(xn,yn|zi)
            if xi_yi is not None:
                p_obs_given_hidden = likelihood_function[xi_yi] #p(xi,yi|zi) were xi,yi is our observation
            else:
                p_obs_given_hidden = 1
            for zi_1 in alpha_zi_1:
                nextgivenprev = transition_model(zi_1)
                sum += alpha_zi_1[zi_1]*nextgivenprev[zi]
            if p_obs_given_hidden*sum>0.0:
                alpha_zi[zi] = p_obs_given_hidden * sum

        alpha_zi.renormalize()
        forward_messages[i] = alpha_zi


    # TODO: Compute the backward messages

    #Initialize the last beta

    beta_zN_1 = prior_distribution
    for state in all_possible_hidden_states:
        beta_zN_1[state] = 1
    beta_zN_1.renormalize()
    backward_messages[-1] =  beta_zN_1

    #Recursively go back for other betas
    beta_zn_1 = beta_zN_1

    print("COMPUTING BETA VALUES NOW")
    for n in range(num_time_steps-2, -1, -1):


        beta_zn = backward_messages[n+1]
        beta_zn_1 = rover.Distribution()
        xn_yn = observations[n + 1]
        #print(xn_yn)

        for zn_1 in all_possible_hidden_states:
            sum = 0
            nextgivenprev = transition_model(zn_1)
            for zn in beta_zn:

                likelihood_function = observation_model(zn)
                if xn_yn is not None:
                    p_obs_given_hidden = likelihood_function[xn_yn]
                else:
                    p_obs_given_hidden = 1
                sum += beta_zn[zn]*p_obs_given_hidden*nextgivenprev[zn]
            if sum > 0.0:
                beta_zn_1[zn_1] = sum
        #print(beta_zn_1)
        beta_zn_1.renormalize()
        backward_messages[n] = beta_zn_1


    # TODO: Compute the marginals
    print("PRINTING MARGINALS")
    for i in range(num_time_steps):
        alpha_zi = forward_messages[i]
        beta_zi = backward_messages[i]
        gamma_zi = rover.Distribution()
        for state in all_possible_hidden_states:
            if alpha_zi[state]*beta_zi[state] > 0:
                gamma_zi[state] = alpha_zi[state]*beta_zi[state]
        gamma_zi.renormalize()


        marginals[i] = gamma_zi

        print("Time step: {} | marginal = {}".format(i, marginals[i]))
    return marginals

def mlstate(x):
    maximum = -np.inf
    mostlikely = None

    for state in x:
        if x[state] > maximum:
            mostlikely = state
            maximum = x[state]
    return mostlikely, maximum


def mle_past(Zn_1, W_prev):
    max_log_prob = -np.inf
    arg_max = None
    for prev_state in W_prev:
        curr = logarithm(rover.transition_model(prev_state)[Zn_1]) + W_prev[prev_state]
        if curr > max_log_prob:
            max_log_prob = curr
            arg_max = prev_state
    return max_log_prob, arg_max


def logarithm(x):
    if x == 0:
        return -np.inf
    else:
        return np.log(x)

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here


    logpxcondz = rover.Distribution()
    w1 = rover.Distribution()
    for z1 in prior_distribution:
        likelihood = observation_model(z1)
        if (observations[0] is not None):
            logpxcondz[z1] = logarithm(likelihood[observations[0]])
        w1[z1] = logpxcondz[z1] + logarithm(prior_distribution[z1])

    W = [None] * len(observations)
    W[0] = w1

    parent = []
    for i in range(len(observations)):
        parent.append({})


    print("CALCULATING W VALUES")
    for n in range(1, len(observations)):

        wn = rover.Distribution()
        logpxcondz = rover.Distribution()

        for hidden_state in all_possible_hidden_states:
            pxcondz = observation_model(hidden_state)

            if (observations[n] is not None):
                logpxcondz[hidden_state] = logarithm(pxcondz[observations[n]])

            max_log_prob, MLprev = mle_past(hidden_state, W[n - 1])

            if MLprev is not None:
                parent[n][hidden_state] = MLprev
                wn[hidden_state] = logpxcondz[hidden_state] + max_log_prob

        W[n] = wn


    end = W[-1]

    most_likely_state, max_log_prob = mlstate(end)

    state_sequence = []
    state_sequence.append(most_likely_state)
    for n in range(len(observations) - 1, 0, -1):

        past = state_sequence[-1]
        present = parent[n][past]
        state_sequence.append(present)


        print("Time step = {} | Joint(t) = {}".format(n, present))

    estimated_hidden_states = state_sequence[::-1]

    return estimated_hidden_states

def bool_list_isequal(L1, L2):

    if len(L1) != len(L2):
        print("ERROR")
        return -1

    length = len(L1)
    bool_list =[None]*length
    for i in range(length):
        bool_list[i] = (L1[i] == L2[i])

    return bool_list
if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
  
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps

    ml_marginal = [None]*num_time_steps
    for i in range(num_time_steps):
        mlhidden, prob = mlstate(marginals[i])
        ml_marginal[i] = mlhidden

    state_sequence_array = np.array(bool_list_isequal(estimated_states, hidden_states))
    marginal_array = np.array(bool_list_isequal(ml_marginal, hidden_states))


    pct_correct = state_sequence_array.sum()/num_time_steps
    p_e_ss = 1-pct_correct

    pct_correct = marginal_array.sum()/num_time_steps
    p_e_marg = 1-pct_correct

    print("ML Marginal Error {} | ML Sequence Error {}".format(p_e_marg, p_e_ss))
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()

