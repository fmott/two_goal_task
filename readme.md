# Data and code

## Dynamic integration of forward planning and heuristic preferences during multiple goal pursuit

**Florian Ott<sup>1\*</sup>, Dimitrije Markovic<sup>1</sup>, Alexander Strobel<sup>1</sup>, Stefan J. Kiebel<sup>1</sup>**

<sup>1</sup>Department of Psychology, Technische Universität Dresden, Germany 
\* Corresponding author
E-Mail: florian.ott@tu-dresden.de

The repository contains two main folders:

* Results
  * *preprocessed_results.csv*
  * *preprocessed_optimal_agent.csv*
  * theta_beta_gamma_kappa
  * elbo
* Code

### Results

The file *preprocessed_results.csv* contains behavioural data of 89 participants. The file *preprocessed_optimal_agent.csv* contains simulated data of 100 instances of an optimal agent. Columns are:

**response:** 1 = 'accept', 2 = 'wait' 
**rt:** reaction time in seconds												
**score_A_before:** A-points (Pts<sup>A</sup>) +1 before response
Note that e.g. a value of 5 in the data corresponds to 4 points in the experiment because 0 points in the data is denoted as state 1.
**score_B_before:** B-points (Pts<sup>B</sup>) + 1 before response
**score_A_after:** A-points (Pts<sup>A</sup>) +1 after response
**score_B_after:** B-points (Pts<sup>B</sup>) +1 after response
**offer:** 1 = A, 2 = B, 3 = Ab, 4 = aB
**start_condition:** 1 = (5,7); 2 = (7,5); 3 = (6,8); 4 = (8,6); 5 = (7,9); 6 = (9,7)
The first number in parentheses denotes Pts<sup>A</sup> + 1 and the second number Pts<sup>B</sup> + 1 in the first trial.
**trial:** 1-15 trials in a miniblock 
**block:** 1-10 miniblocks in the training phase and 1-20 miniblocks in each main experimental phase.
The main experimental phase was subdivided into three phases between which participants could pause.
**phase:**  1 = training, 2 - 4 = main experimental phase
**subject:** 1 - 89
**timeout:** participant did not respond in time
**valid:** no timeout and not both goals reached 
**dv:** expected future reward of choice 'accept' minus choice 'wait' (&beta; -> &infin;, &theta; = 0, &gamma; = 1, &kappa; = 1) 
**score_difference**: A- minus  B-points 
**goal_drive:** expected future reward of choice 'g2' minus choice 'g1' (&beta; -> &infin;, &theta; = 0, &gamma; = 1, &kappa; = 1)
This corresponds to differential expected value (DEV) in the manuscript for an optimal parametrization. 
**goal_decision:** 2 = g2-choice, 1 = g1-choice
Called goal strategy choice in the manuscript 
**suboptimal_decision:** response deviates from optimal agent
**suboptimal_goal_decision:**  goal decision deviates from optimal agent; 1 = optimal g1, 2 = optimal g2, -1 = suboptimal g2, -2 = suboptimal g1 
**reported_strategy:** 1 = sequential, 2 = parallel, 3 = mixed
**gender:** 0 = male, 1 = female 
**age:** in years

The sub-folder theta_beta_gamma_kappa contains model results:

* *elbo_20191022_1411_45.csv* contains the trace of (negative) evidence lower bound (- Elbo).

* *group_posterior_sample_20191022-141145.csv* contains 1,000 samples from the group posterior over parameters (&beta;, &theta;, &gamma;, &kappa;) in unconstrained space.

* *subject_posterior_sample_20191022-141145.csv* contains 1,000 samples from the subject-specific posterior over parameters (&beta;, &theta;, &gamma;, &kappa;) in unconstrained space.

* *posterior_median_percentiles_20191022-141145.csv* contains 5%, 50% and 95% quantiles of subject specific posterior in constrained space. The labels in the parameter column correspond to (beta_2 = &beta;, bias_2 = &theta;, gamma = &gamma;, kappa = &kappa;).

The sub-folder elbo contains elbo traces of different model variants used to compare models.

### Code

**tg_fit.py:** fits the model. 
**tg_fit_split_cond.py:** fits the model separately for the the easy, medium and hard difficulty condition.
**tg_fit_split_phase.py:** fits the model separately for the experimental phases. 
**tg_fit_split_seg.py:** fits the model separately for miniblock segments (trial 1-5, trial 6-10, trial 11-15).

**agents.py:** class *Informed*  defines the forward planning agent.
**helpers.py:** function *offer_state_mapping* defines the transition matrix
**inference.py:** coordinates the interaction between different modules that govern the agent's behavior
**simulate.py:** class *Simulato*r defines interactions between the environment and the agent
**tasks.py:** defines the task environment

**tg_analysis.py:** calls functions to process data and generate figures 3, 4, 5, 6 and S5, S8, S12
**tg_simulations.py:** simulates a random agent and generates supplementary figures S6-S7; performs posterior predictive simulations generating figures S10-S11; simulates data with varying &beta; or &theta; and generates frames for movies S1-S4. 
**tg_compare_elbo.py:** compares elbo of model variants and generates figure S9. 
**tg_offer_stats.py:** generates figures S1-S4 providing information about the offer sequence

**Testing inference.ipynb:** Jupyter notebook (S1 Notebook) performing parameter recovery simulations 

**tg_performance_phase_mlm.R:** linear mixed effects model testing for learning effects 
**tg_subg_phase_mlm.R:** logistic mixed effects model testing for learning effects
**tg_gc_scorediff_miniblockhalf.R:** logistic regression testing for the use of a heuristic depending on miniblock half. 