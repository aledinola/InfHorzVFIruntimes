clear,clc,close all
addpath(genpath('C:\Users\aledi\Documents\GitHub\VFIToolkit-matlab'))
% Do pure discretized VFI for simply problem
% If n_z=2, this is the model of Rendahl (2022)
% Otherwise, is roughly the same value fn problem as Aiyagari (1994) [minor differences]
%
% Purpose of this code is to understand how VFI Toolkit should set defaults
% for Howards iteration vs Howards Greedy (a.k.a., modified-Policy fn
% iteration vs Policy fn iteration).
% For Howards iteration, there are two versions coded, the first doing the
% expections based on indexing a matrix, the second uses a sparse matrix
% multiplication.
% I also did a 'hardcode' of the interals of VFI Toolkit commands, to see
% if overhead was an issue.
%
% Result of all this is:
% Howards-greedy is fastest for very small problems, VFI Toolkit defaults has been set to use greedy when N_a<400 || N_z<20
% For anything but very small problems, VFI Toolkit uses Howards iteration
% Howards iteration is faster with indexing that spare matrix, for expectations.
% Overhead costs of ValueFnIter command versus the hardcode is negligible.
% [Not covered here, but by default VFI Toolkit would 'refine' away a 'd' (decision) variable. 
% Then based on N_a and N_z will use Howards greedy or iteration on what is left]
%
% Note: I report runtimes, and comparison to those of Rendahl (2022) for
% the model with n_z=2, and n_a=500,1000,1500 (which are what is in the
% paper). Rendahl (2022) uses MCA, so there is no expectation that codes
% here get equally fast, is just used out of interest.
%
% For discussion of VFI, Howards-iteration, Howards-greedy, etc., see
% http://discourse.vfitoolkit.com/t/vfi-in-infhorz-howards-pfi-and-relation-to-implicit-finite-differences/408

%% Set some basic variables

% Size of the grids
n_d=0;
n_a=5;
n_z=2;

% Parameters
Params.gamma = 2; % CRRA coeff in preferences
Params.beta  = 0.96; % discount rate, is over 0.99
Params.r     = 0.04; % interest rate [what Rendahl has in eqm]

%% Grids
d_grid=[]; %There is no d variable
% Set grid for asset holdings
Params.amax=400; % took this from Rendahl codes, he has evenly spaced points from 0 to amax, and amax=400
a_grid=Params.amax*linspace(0,1,n_a)'; % evenly spaced, not a good idea

%%
DiscountFactorParamNames={'beta'};

ReturnFn = @(aprime,a,z,r,w,gamma) IFP_ReturnFn(aprime,a,z,r,w,gamma);
% The first inputs must be: next period endogenous state, endogenous state, exogenous state. Followed by any parameters

vfoptions=struct();

Tolerance=10^(-9);
maxiter=Inf;
maxhowards=500; % just a safety valve on max number of times to do Howards, not sure it is needed for anything?

%% Fix N_a and N_z

fprintf('Currently doing n_a=%d, n_z=%d \n',n_a,n_z)

a_grid=gpuArray(a_grid);
z_gridvals=gpuArray(z_gridvals);
pi_z=gpuArray(pi_z);

DiscountFactorParamsVec=Params.beta;
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Params);
ReturnFnParamsVec=CreateVectorFromParams(Params, ReturnFnParamNames);

ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_nod_Par2(ReturnFn, n_a, n_z, a_grid, z_gridvals, ReturnFnParamsVec);

N_a=prod(n_a);
N_z=prod(n_z);

pi_z_alt=shiftdim(pi_z',-1);

addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

V0=zeros(N_a,N_z,'gpuArray');

           

            
            clear pi_z_howards

            % [setuptime,Rendahltest1]
            % tempcounter1

            VKron_Hiter=VKron;

            
            % [setuptime,Rendahltest2]
            % tempcounter2

            VKron_Hiter2=VKron;

            %% Third, greedy Howards, so as a linear system of equations
            VKron=V0;

            tic;
            % Setup specific to greedy Howards
            spI = speye(N_a*N_z);
            T_pi_z=sparse(repelem(pi_z,N_a,N_a)); % row is this period, column is next period: (a,z) to (a',z')

            tempcounter3=1;
            currdist=Inf;
            while currdist>Tolerance && tempcounter3<=maxiter
                VKronold=VKron;

                % Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV=VKronold.*pi_z_alt;
                EV(isnan(EV))=0; % multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV=sum(EV,2); % sum over z', leaving a singular second dimension

                entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; %aprime by a by z

                %Calc the max and it's index
                [VKron,Policy]=max(entireRHS,[],1);
                VKron=shiftdim(VKron,1); % a by z

                VKrondist=VKron(:)-VKronold(:);
                VKrondist(isnan(VKrondist))=0;
                currdist=max(abs(VKrondist));

                % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
                if isfinite(currdist) && currdist/Tolerance>10 && tempcounter3<maxhowards
                    tempmaxindex=shiftdim(Policy,1)+addindexforaz; % aprime index, add the index for a and z
                    Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

                    T_E=sparse(repelem((1:1:N_a*N_z)',1,N_z),Policy(:)+N_a*(0:1:N_z-1),repelem(pi_z,N_a,1),N_a*N_z,N_a*N_z);

                    VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
                    VKron=reshape(VKron,[N_a,N_z]);
                end

                tempcounter3=tempcounter3+1;

            end

            Policy=reshape(Policy,[N_a,N_z]);

            Rendahltest3=toc;


            % [setuptime,Rendahltest3]
            % tempcounter3

            VKron_Hgreedy=VKron;

            %% Check same solution
            checkzero(a_c,z_c,h_c,1)=max(abs(VKron_Hiter(:)-VKron_Hiter2(:)));
            checkzero(a_c,z_c,h_c,2)=max(abs(VKron_Hiter(:)-VKron_Hgreedy(:)));

            % Store runtimes
            setuptimes(a_c,z_c,h_c)=setuptime;
            runtimes(a_c,z_c,h_c,1)=Rendahltest1;
            runtimes(a_c,z_c,h_c,2)=Rendahltest2;
            runtimes(a_c,z_c,h_c,3)=Rendahltest3;

            % Store counters
            counter(a_c,z_c,h_c,1)=tempcounter1;
            counter(a_c,z_c,h_c,2)=tempcounter2;
            counter(a_c,z_c,h_c,3)=tempcounter3;

        end
    end
end

%% Look at results

% First, just check we get same solutions
max(max(max(abs(checkzero(:,:,:,1))))) % zero. Good
max(max(max(abs(checkzero(:,:,:,2))))) % close to zero, good enough

% Average runtimes across everything
max(max(max(abs(setuptimes(:,:,:))))) % this is a tiny fraction of the time
max(max(max(abs(runtimes(:,:,:,1))))) % Fastest on average
max(max(max(abs(runtimes(:,:,:,2)))))
max(max(max(abs(runtimes(:,:,:,3))))) % Greedy is slower most of the time

% Is one of the two iteration implementations always better than the other?
min(min(min(abs(runtimes(:,:,:,1)./runtimes(:,:,:,2))))) % 0.73
max(max(max(abs(runtimes(:,:,:,1)./runtimes(:,:,:,2))))) % 0.83
% So runtime1/runtime2 ranges from 0.73 to 0.83, which is always less than
% one. So first implementation (using indexes) is faster than the second
% (using sparse matrix and multiplication).

% What about the best H?
[~,Hoptindex]=min(runtimes(:,:,:,1),[],3); % third dimension indexes H
[min(Hoptindex(:)),max(Hoptindex(:))]
% always 4 to 7,
% which corresponds to H=80 to 150
Hoptindex
% no obvious rule.
% What is average using H=150 vs 80 or 100
temp4=runtimes(:,:,4,1);
temp5=runtimes(:,:,5,1);
temp6=runtimes(:,:,6,1);
temp7=runtimes(:,:,7,1);
[mean(temp4(:)),mean(temp5(:)),mean(temp6(:)),mean(temp7(:))] % minor differences at most
[min(temp4(:)./temp6(:)), max(temp4(:)./temp6(:))] % 4 is 7% better to 11% worse (than 6, H=150)
[min(temp5(:)./temp6(:)), max(temp5(:)./temp6(:))] % 5 is 7% better to 14% worse (than 6, H=150)
[min(temp6(:)./temp6(:)), max(temp6(:)./temp6(:))] % 
[min(temp7(:)./temp6(:)), max(temp7(:)./temp6(:))] % 7 is 8% better to 23% worse (than 6, H=150)
% So nothing ever beat H=150 by more than a few percent
% And H=150 on average


%% When is iter with index better than iter with sparse
index=((runtimes(:,:,:,1)./runtimes(:,:,:,2))<1)
% Always better

%% When is Greedy best?
greedy=((runtimes(:,:,:,3)./runtimes(:,:,:,1))<1)
% Greedy is better all except for n_a=500 with n_z>=15

%% Decision: do a check "if N_a<400 || N_z<20" send to Howards greedy, "else" send to Howards iteration
% And always use Howards iteration with indexing, not with sparse matrix





%% Compare to get Rendahl (2022) times by just dropping the 'safety' checks on EV being NaN and on Howards avoiding -Inf
Rendahl_N_a_vec=[500,1000,1500];
% Roughly, takes me 0.14, 0.22, 0.33s, respectively
% Rendahl reports 0.02, 0.04, 0.04 for discrete time
% Rendahl reports 0.02, 0.07, 0.13 for continuous time
% Disabling the 'safety' checks has essentially zero impact on run times
% Of course Rendahl is just doing 'local policy search/MCA', so that should be much faster as checking way fewer points.

n_z=2;
for a_c=1:length(Rendahl_N_a_vec)
    n_a=Rendahl_N_a_vec(a_c);

    ReturnFn=@(aprime, a, z,r,alpha,delta,mu,tau,gamma) Rendahl2022_ReturnFn(aprime, a, z,r,alpha,delta,mu,tau,gamma);

    z_gridvals=[0;1];
    pi_z=[1-Params.phi, Params.phi; Params.delta, 1-Params.delta];
    a_grid=Params.amax*linspace(0,1,n_a)'; % evenly spaced, not a good idea


    %% First, just some copy-paste of toolkit internals to get things setup
    tic;

    n_a=gpuArray(n_a);
    n_z=gpuArray(n_z);
    a_grid=gpuArray(a_grid);
    z_gridvals=gpuArray(z_gridvals);
    pi_z=gpuArray(pi_z);

    DiscountFactorParamsVec=Params.beta;
    ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Params);
    ReturnFnParamsVec=CreateVectorFromParams(Params, ReturnFnParamNames);

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_nod_Par2(ReturnFn, n_a, n_z, a_grid, z_gridvals, ReturnFnParamsVec);

    N_a=prod(n_a);
    N_z=prod(n_z);

    pi_z_alt=shiftdim(pi_z',-1);

    addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

    V0=zeros(N_a,N_z,'gpuArray');

    greedyHind1=gpuArray(repelem((1:1:N_a*N_z)',1,N_z));
    greedyHind2=gpuArray(N_a*(0:1:N_z-1));
    greedyHpi=gpuArray(repelem(pi_z,N_a,1));
    % % Preallocate. Does nothing
    % Policy=ones(size(V0),'gpuArray');
    % T_E=sparse(greedyHind1,Policy(:)+greedyHind2,greedyHpi,N_a*N_z,N_a*N_z);


    setuptime=toc;

    %% Third, greedy Howards, so as a linear system of equations
    VKron=V0;

    tic;
    % Setup specific to greedy Howards
    N_az=N_a*N_z;
    spI = gpuArray.speye(N_az);
    T_pi_z=sparse(repelem(pi_z,N_a,N_a)); % row is this period, column is next period: (a,z) to (a',z')

    tempcounter3=1;
    currdist=Inf;
    while currdist>Tolerance
        VKronold=VKron;

        % Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV=VKronold.*pi_z_alt;
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; %aprime by a by z

        %Calc the max and it's index
        [VKron,Policy]=max(entireRHS,[],1);
        VKron=shiftdim(VKron,1); % a by z (I SUSPECT THIS CAN BE DROPPED, JUST DO AFTER LOOP OR IF NOT USING HOWARDS

        VKrondist=VKron(:)-VKronold(:);
        VKrondist(isnan(VKrondist))=0;
        currdist=max(abs(VKrondist));

        % Use greedy-Howards Improvement (except for first few and last few iterations, as it is not a good idea there)
        tempmaxindex=shiftdim(Policy,1)+addindexforaz; % aprime index, add the index for a and z
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards

        T_E=sparse(greedyHind1,Policy(:)+greedyHind2,greedyHpi,N_a*N_z,N_a*N_z);

        VKron=(spI-DiscountFactorParamsVec*T_E)\Ftemp;
        VKron=reshape(VKron,[N_a,N_z]);

        tempcounter3=tempcounter3+1;

    end

    Policy=reshape(Policy,[N_a,N_z]);

    Rendahltest3=toc;

    [n_a,tempcounter3]
    [setuptime+Rendahltest3,setuptime,Rendahltest3]


end


%% Lastly, redo Rendahl (2022), but now with full VFI Toolkit overhead, see how that changes the times
n_z=2;
for a_c=1:length(Rendahl_N_a_vec)
    n_a=Rendahl_N_a_vec(a_c);

    ReturnFn=@(aprime, a, z,r,alpha,delta,mu,tau,gamma) Rendahl2022_ReturnFn(aprime, a, z,r,alpha,delta,mu,tau,gamma);

    z_gridvals=[0;1];
    pi_z=[1-Params.phi, Params.phi; Params.delta, 1-Params.delta];
    a_grid=Params.amax*linspace(0,1,n_a)'; % evenly spaced, not a good idea

    tic;
    [V,Policy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_gridvals,pi_z,ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
    Rendahltest4=toc
end
% takes 0.13, 0.2 and 0.31 seconds 
% So overhead is trivial








