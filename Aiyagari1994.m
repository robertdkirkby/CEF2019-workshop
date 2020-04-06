% Example based on Aiyagari (1994). 
% Follows the nomenclature of (k,l) used by Aiyagari (1994) to denote
% capital and (exogenous) labor.

% These codes do some things that take quite a long time and are not really
% necessary (the MarkovChainMoments). They are included for completeness.

% These codes set up and solve the Aiyagari (1994) model for a given
% parametrization. After solving the model they then show how some of the
% vfitoolkit commands to easily calculate things like the Gini coefficient
% for income, and how to plot the distribution of asset holdings.

% Grid sizes, have declared at beginning for convenience.
n_k=2^7;%2^9;
n_l=15; %21;
n_r=0; % Normally you will want n_p=0, setting a non-zero value here activates the use of a grid on prices.

% Parallel=2 for GPU, Parallel=1 is parallel CPU, Parallel=0 is single CPU
Parallel=2;

%% Declare the model parameters (keep them all in a structure Params)
%Parameters
Params.beta=0.96; % Discount factor
Params.alpha=0.36; % Capital share in Cobb-Douglas Production function
Params.delta=0.08; % Depreciation rate of capital
Params.mu=3; % CRRA parameter in utility function
Params.rho=0.6; % Autocorrelation of z
Params.sigma=0.2; % Std dev. of shocks to z

%Set initial value for interest rates (Aiyagari proves that with idiosyncratic
%uncertainty, the eqm interest rate is limited above by it's steady state value
%without idiosyncratic uncertainty, that is that r<r_ss).
Params.r=0.04;

Params.q=3; %Footnote 33 of Aiyagari(1993WP, pg 25) implicitly says that he uses q=3

%% Some Toolkit options (most of these are anyway just being set to toolkit defaults)
tauchenoptions.parallel=Parallel;
mcmomentsoptions.parallel=Parallel;

vfoptions.lowmemory=0;
vfoptions.parallel=Parallel;

simoptions.burnin=10^4;
simoptions.simperiods=10^5; % For an accurate solution you will either need simperiod=10^5 and iterate=1, or simperiod=10^6 (iterate=0).
simoptions.iterate=1;
simoptions.parallel=Parallel;

heteroagentoptions.verbose=1; % verbose=1 means give feedback/output, verbose=0 means run largely silent.

%% Set up the exogenous shock process
% Create markov process for the exogenous labour productivity, l.
[l_grid, pi_l]=TauchenMethod(0,(Params.sigma^2)*(1-Params.rho^2),Params.rho,n_l,Params.q,tauchenoptions); 
l_grid=exp(l_grid);
% Get some info on the markov process
[Expectation_l,~,~,~]=MarkovChainMoments(l_grid,pi_l,mcmomentsoptions); %Since z is exogenous, this will be it's eqm value 
% Note: Aiyagari (1994) actually then normalizes z by dividing it by
% Expectation_l (so that the resulting process has expectaion equal to 1)
% In theory this would always be true, but for smaller grids on l, it fails
% to be true due to numerical error.
l_grid=l_grid./Expectation_l;

% Calculate some things we will want later (about how the Tauchen method is
% performing as an approximation of the exogenous shock process (Aiyagari,
% 1994, Table 1).
[l_mean,l_variance,l_corr,~]=MarkovChainMoments(l_grid,pi_l,mcmomentsoptions);


%% Grids

% In the absence of idiosyncratic risk, the steady state equilibrium is given by
r_ss=1/Params.beta-1;
K_ss=((r_ss+Params.delta)/Params.alpha)^(1/(Params.alpha-1)); %The steady state capital in the absence of aggregate uncertainty.

% Set grid for asset holdings
nk1=floor(n_k/3); nk2=floor(n_k/3); nk3=n_k-nk1-nk2;
k_grid=sort([linspace(0,K_ss,nk1),linspace(K_ss+0.0001,3*K_ss,nk2),linspace(3*K_ss+0.0001,15*K_ss,nk3)]');

% Bring model into the notational conventions used by the toolkit
% To simplify this example I have used the (a,z) notation of the VFI
% Toolkit directly.
n_d=0;
d_grid=0; %There is no d variable
n_a=n_k;
a_grid=k_grid;
n_z=n_l;
z_grid=l_grid;
pi_z=pi_l;
n_p=n_r;

%% STEP 3: Solving the value function problem
DiscountFactorParamNames={'beta'};

ReturnFn=@(aprime_val, a_val, z_val,alpha,delta,mu,r) Aiyagari1994_ReturnFn(aprime_val, a_val, z_val,alpha,delta,mu,r);
ReturnFnParamNames={'alpha','delta','mu','r'}; %It is important that these are in same order as they appear in 'Aiyagari1994_ReturnFn'

% Following lines could be used to test that we are setting things up
% correctly, but is not needed at this stage.
% [V,Policy]=ValueFnIter_Case1(n_d,n_k,n_l,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
% StationaryDist=StationaryDist_Case1(Policy,n_d,n_k,n_l,pi_z);

%% Solve for the General Equilibrium

% Create descriptions of aggregate values as functions of d_grid, a_grid, z_grid 
% (used to calculate the integral across the stationary dist fn of whatever functions you define here)
FnsToEvaluateParamNames.Names={};
FnsToEvaluate_1 = @(aprime,a,z) a; %We just want the aggregate assets (which is this periods state)
FnsToEvaluate={FnsToEvaluate_1};

% Following line could be used to test that we are setting things up correctly, but is not needed at this stage.
AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDist, Policy, FnsToEvaluate, Params, FnsToEvaluateParamNames, n_d, n_k, n_l, d_grid, a_grid, z_grid, Parallel);

%Now define the functions for the General Equilibrium conditions
    %Should be written as LHS of general equilibrium eqn minus RHS, so that 
    %the closer the value given by the function is to zero, the closer 
    %the general equilibrium condition is to holding.
% Note AggVars contains the expected values over the stationary agent
% distribution of the FnsToEvaluate
GeneralEqmEqnsParamNames(1).Names={'alpha','delta'};
GeneralEqmEqn_1 = @(AggVars,p,alpha,delta) p-(alpha*(AggVars^(alpha-1))*(Expectation_l^(1-alpha))-delta); %The requirement that the interest rate corresponds to the agg capital level
GeneralEqmEqns={GeneralEqmEqn_1};

%Use the toolkit to find the equilibrium price index
PriceParamNames={'r'};

disp('Calculating price vector corresponding to the stationary eqm')
[p_eqm,~,GeneralEqmCondns]=HeteroAgentStationaryEqm_Case1(n_d, n_k, n_l, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnsParamNames, PriceParamNames,heteroagentoptions, simoptions, vfoptions);
% The three output are the general equilibrium price, the index for the
% price in the price grid (that option is unused here), and the value of
% the General equilibrium conditions in equilibrium (note that they should
% be zero, or in practice say of the order of 10^(-3) or 10^(-5)).
p_eqm

% % Alternatively, you could use the p_grid option
% n_p=101; p_grid=linspace(0.03,r_ss,n_p); heteroagentoptions.pgrid=p_grid;
% [p_eqm2,p_eqm_index2,MarketClearanceVec2]=HeteroAgentStationaryEqm_Case1(n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnsParamNames, PriceParamNames,heteroagentoptions, simoptions, vfoptions);
% % Can even do so as multigrid (use course grid, then fine grid in region of solution)
% p_grid2=linspace(p_grid(p_eqm_index2-5),p_grid(p_eqm_index2+5),n_p); heteroagentoptions.pgrid=p_grid2;
% [p_eqm3,p_eqm_index3,MarketClearanceVec3]=HeteroAgentStationaryEqm_Case1(n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnsParamNames, PriceParamNames,heteroagentoptions, simoptions, vfoptions);
% 
% [p_eqm.r, p_eqm2.r, p_eqm3.r]

%% Now that we have the GE, let's calculate a bunch of related objects

% Equilibrium wage
Params.w=(1-Params.alpha)*((p_eqm.r+Params.delta)/Params.alpha)^(Params.alpha/(Params.alpha-1));

disp('Calculating various equilibrium objects')
Params.r=p_eqm.r;
[V,Policy]=ValueFnIter_Case1(n_d,n_k,n_l,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

% By default Policy contains the indexes corresponding to the optimal
% policy. Can get the policy values using vfoptions.polindorval=1 or,
% PolicyValues=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid,a_grid, Parallel);

StationaryDist=StationaryDist_Case1(Policy,n_d,n_k,n_l,pi_z, simoptions);

AggregateVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDist, Policy, FnsToEvaluate,Params, FnsToEvaluateParamNames,n_d, n_k, n_l, d_grid, a_grid,z_grid, Parallel)

% save ./SavedOutput/Aiyagari1994_Objects.mat p_eqm Policy StationaryDist

% Calculate savings rate:
% We know production is Y=K^{\alpha}L^{1-\alpha}, and that L=1
% (exogeneous). Thus Y=K^{\alpha}.
% In equilibrium K is constant, so aggregate savings is just depreciation, which
% equals delta*K. The agg savings rate is thus delta*K/Y.
% So agg savings rate is given by s=delta*K/(K^{\alpha})=delta*K^{1-\alpha}
aggsavingsrate=Params.delta*AggregateVars^(1-Params.alpha);

% Calculate Lorenz curves, Gini coefficients, and Pareto tail coefficients
%  @(d_val,aprime_val,a_val,s_val,pi_z,p_val,param)
FnsToEvaluateParamNames(1).Names={'w'};
FnsToEvaluate_Earnings = @(aprime_val,a_val,z_val,w) w*z_val;
FnsToEvaluateParamNames(2).Names={'w','r'};
FnsToEvaluate_Income = @(aprime_val,a_val,z_val,w,r) w*z_val+(1+r)*a_val;
FnsToEvaluateParamNames(3).Names={};
FnsToEvaluate_Wealth = @(aprime_val,a_val,z_val) a_val;
FnsToEvaluateIneq={FnsToEvaluate_Earnings, FnsToEvaluate_Income, FnsToEvaluate_Wealth};
LorenzCurves=EvalFnOnAgentDist_LorenzCurve_Case1(StationaryDist, Policy, FnsToEvaluateIneq, Params,FnsToEvaluateParamNames, n_d, n_k, n_l, d_grid, a_grid, z_grid, Parallel);

% 3.5 The Distributions of Earnings and Wealth
%  Gini for Earnings
EarningsGini=Gini_from_LorenzCurve(LorenzCurves(1,:));
IncomeGini=Gini_from_LorenzCurve(LorenzCurves(2,:));
WealthGini=Gini_from_LorenzCurve(LorenzCurves(3,:));

% Calculate inverted Pareto coeff, b, from the top income shares as b=1/[log(S1%/S0.1%)/log(10)] (formula taken from Excel download of WTID database)
% No longer used: Calculate Pareto coeff from Gini as alpha=(1+1/G)/2; ( http://en.wikipedia.org/wiki/Pareto_distribution#Lorenz_curve_and_Gini_coefficient)
% Recalculte Lorenz curves, now with 1000 points
LorenzCurves=EvalFnOnAgentDist_LorenzCurve_Case1(StationaryDist, Policy, FnsToEvaluateIneq, Params,FnsToEvaluateParamNames, n_d, n_k, n_l, d_grid, a_grid, z_grid, Parallel,1000);
EarningsParetoCoeff=1/((log(LorenzCurves(1,990))/log(LorenzCurves(1,999)))/log(10)); %(1+1/EarningsGini)/2;
IncomeParetoCoeff=1/((log(LorenzCurves(2,990))/log(LorenzCurves(2,999)))/log(10)); %(1+1/IncomeGini)/2;
WealthParetoCoeff=1/((log(LorenzCurves(3,990))/log(LorenzCurves(3,999)))/log(10)); %(1+1/WealthGini)/2;


%% Display some output about the solution

plot(a_grid,cumsum(sum(StationaryDist,2))) %Plot the asset cdf

fprintf('For parameter values sigma=%.2f, mu=%.2f, rho=%.2f \n', [Params.sigma,Params.mu,Params.rho])
fprintf('The table 1 elements are sigma=%.4f, rho=%.4f \n',[sqrt(l_variance), l_corr])

fprintf('The equilibrium value of the interest rate is r=%.4f \n', p_eqm*100)
fprintf('The equilibrium value of the aggregate savings rate is s=%.4f \n', aggsavingsrate)
%fprintf('Time required to find the eqm was %.4f seconds \n',findeqmtime)

