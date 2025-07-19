function [outputArg1,outputArg2] = fun_VFI_iter_indexing(inputArg1,inputArg2)

%% First, Howards iteration, with H iterations, using index
VKron=V0;

tic;
% Setup specific to Howard iterations
% H=80; % number of Howards iterations
pi_z_howards=repelem(pi_z,N_a,1);

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
        Ftemp=reshape(ReturnMatrix(tempmaxindex),[N_a,N_z]); % keep return function of optimal policy for using in Howards
        Policy=Policy(:); % a by z (this shape is just convenient for Howards)

        for Howards_counter=1:H
            EVKrontemp=VKron(Policy,:);
            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VKron=Ftemp+DiscountFactorParamsVec*EVKrontemp;
        end
    end

    tempcounter1=tempcounter1+1;

end

Policy=reshape(Policy,[N_a,N_z]);

end %end function