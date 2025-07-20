clear,clc,close all
%addpath(genpath('C:\Users\aledi\Documents\GitHub\VFIToolkit-matlab'))
addpath(genpath('C:\Users\aledi\OneDrive\Documents\GitHub\VFIToolkit-matlab'))

%% Set some basic variables

% Size of the grids
n_d=0;
n_a=1000;
n_z=21;

% Parameters
Params.gamma = 2; % CRRA coeff in preferences
Params.beta  = 0.96; % discount rate, is over 0.99
Params.r     = 0.04; % interest rate [what Rendahl has in eqm]
Params.w     = 1.0;
Params.rho   = 0.9;  % persistence AR1
Params.sigma = 0.1; % standard deviation iid shock


%% Grids
d_grid=[]; %There is no d variable
% Set grid for asset holdings
Params.amax=400; % took this from Rendahl codes, he has evenly spaced points from 0 to amax, and amax=400
a_grid=(Params.amax*linspace(0,1,n_a).^3)'; % evenly spaced, not a good idea

[z_grid,pi_z]=discretizeAR1_Tauchen(0.0,Params.rho,Params.sigma,n_z,3.0);
z_grid = exp(z_grid);

%%
DiscountFactorParamNames={'beta'};

ReturnFn = @(aprime,a,z,r,w,gamma) IFP_ReturnFn(aprime,a,z,r,w,gamma);
% The first inputs must be: next period endogenous state, endogenous state, exogenous state. Followed by any parameters

vfoptions=struct();

Tolerance=10^(-9);
maxiter=10000;
maxhowards=500; % just a safety valve on max number of times to do Howards, not sure it is needed for anything?
H = 80;

%% Fix N_a and N_z

fprintf('Currently doing n_a=%d, n_z=%d \n',n_a,n_z)

z_gridvals = z_grid;
a_grid=gpuArray(a_grid);
z_gridvals=gpuArray(z_gridvals);
pi_z=gpuArray(pi_z);

DiscountFactorParamsVec=Params.beta;
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Params);
ReturnFnParamsVec=CreateVectorFromParams(Params, ReturnFnParamNames);

ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_nod_Par2(ReturnFn, n_a, n_z, a_grid, z_gridvals, ReturnFnParamsVec);

N_a=prod(n_a);
N_z=prod(n_z);

V0=zeros(N_a,N_z,'gpuArray');

%% Method 1: Howard with iterations, indexing

tic
[Policy1] = fun_VFI_iter_indexing(V0,pi_z,N_a,N_z,ReturnMatrix,...
    DiscountFactorParamsVec,Tolerance,maxiter,maxhowards,H);
time1=toc;

%% Method 2: Howard with iterations, sparse matrix

tic
[Policy2] = fun_VFI_iter_sparse(V0,pi_z,N_a,N_z,ReturnMatrix,...
    DiscountFactorParamsVec,Tolerance,maxiter,maxhowards,H);
time2=toc;

%% Method 3: Howard greedy with bigcstab
tic
[Policy3] = fun_VFI_iter_bicgstab(V0,pi_z,N_a,N_z,ReturnMatrix,...
    DiscountFactorParamsVec,Tolerance,maxiter,maxhowards,H);
time3=toc;


err2 = max(abs(Policy2-Policy1),[],"all");
err3 = max(abs(Policy3-Policy1),[],"all");

disp('RUNNING TIMES')
fprintf('fun_VFI_iter_indexing: %f \n',time1)
fprintf('fun_VFI_iter_sparse:   %f \n',time2)
fprintf('fun_VFI_iter_bicgstab: %f \n',time3)
fprintf('err2:   %f \n',err2)
fprintf('err3:   %f \n',err3)







