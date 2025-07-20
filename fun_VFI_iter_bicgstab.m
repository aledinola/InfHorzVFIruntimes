function [Policy] = fun_VFI_iter_bicgstab(V0,pi_z,N_a,N_z,ReturnMatrix,...
    DiscountFactorParamsVec,Tolerance,maxiter,maxhowards,H)

%% First, Howards iteration, with H iterations, using index
VKron=V0;

% Precomputations
pi_z_alt=shiftdim(pi_z',-1);
addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

NzVec = gpuArray.colon(1,N_z)';
z_ind = repmat(NzVec,N_a*N_z,1);
bigI = gpuArray(speye(N_a*N_z));

pi_z1 = repmat(pi_z,1,N_a)';
pi_z_howard = pi_z1(:);

tempcounter1=1;
currdist=Inf;
while currdist>Tolerance && tempcounter1<=maxiter
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
    currdist=max(abs(VKrondist))

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter1<maxhowards
        tempmaxindex=shiftdim(Policy,1)+addindexforaz; % aprime index, add the index for a and z
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards
        
        Policy=Policy(:); % [N_a*N_z,1]
        ind = repelem((1:1:N_a*N_z)',N_z,1);
        indp  = repelem(Policy,N_z,1)+(z_ind-1)*N_a;
        % % Q is policy as mapping from (a,z) to (a',z')
        Q = sparse(ind,indp,pi_z_howard,N_a*N_z,N_a*N_z); 
        %T_E=sparse(repelem((1:1:N_a*N_z)',1,N_z),Policy(:)+N_a*(0:1:N_z-1),repelem(pi_z,N_a,1),N_a*N_z,N_a*N_z);
        VKron = bicgstab(bigI-DiscountFactorParamsVec*Q,Ftemp,1e-10,300);
        VKron = reshape(VKron,[N_a,N_z]);
    end

    tempcounter1=tempcounter1+1;

end

Policy=reshape(Policy,[N_a,N_z]);

end %end function