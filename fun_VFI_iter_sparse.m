function [Policy] = fun_VFI_iter_sparse(V0,pi_z,N_a,N_z,ReturnMatrix,...
    DiscountFactorParamsVec,Tolerance,maxiter,maxhowards,H)

%% First, Howards iteration, with H iterations, using index
VKron=V0;

% Precomputations
pi_z_alt=shiftdim(pi_z',-1);
pi_z_transpose = transpose(pi_z);
addindexforaz=gpuArray(N_a*(0:1:N_a-1)'+N_a*N_a*(0:1:N_z-1));

NaVec   = gpuArray.colon(1,N_a)';
NzVec   = gpuArray.colon(1,N_z)';
a_ind = repmat(NaVec,[N_z,1]);
z_ind = repmat(NzVec',[N_a,1]);
z_ind = z_ind(:);
% [a_ind,z_ind] = ndgrid(NaVec,NzVec);
% a_ind = a_ind(:);
% z_ind = z_ind(:);
ind = a_ind+(z_ind-1)*N_a;

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
    currdist=max(abs(VKrondist));

    % Use Howards Policy Fn Iteration Improvement (except for first few and last few iterations, as it is not a good idea there)
    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter1<maxhowards
        tempmaxindex=shiftdim(Policy,1)+addindexforaz; % aprime index, add the index for a and z
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a*N_z,1]); % keep return function of optimal policy for using in Howards
        
        Policy=Policy(:); % [N_a*N_z,1]
        indp  = Policy+(z_ind-1)*N_a;
        % Q is policy as mapping from (a,z) to (a',z)
        Q = sparse(ind,indp,1,N_a*N_z,N_a*N_z); 
        for Howards_counter=1:H
            EV_howard = VKron*pi_z_transpose; % (a',z)
            EV_howard = reshape(EV_howard,[N_a*N_z,1]);
            VKron = Ftemp+DiscountFactorParamsVec*Q*EV_howard;
            VKron = reshape(VKron,[N_a,N_z]);
        end
    end

    tempcounter1=tempcounter1+1;

end

Policy=reshape(Policy,[N_a,N_z]);

end %end function