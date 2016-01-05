function [sll,V1,V2,dV1,dV2,ll1,ll2,prob1,prob2,s1,s2,r,state2]=modelFreeRL(s1,s2, r, par,V1,V2 )
%V1 (1x2) and V2 (1x4) are values at stage 1 (2 states) and stage 2 (4 states)
%s1 (nx1) is the stage 1 chosen/assigned, [1,2]
%s2 (nx1) is the stage 2 chosen/assigned, [3,4,5,6]
%r (nx1) is the rewards given per round or (nx4) the reward probabilities
%ex: [sll,V1,V2,dV1,dV2,ll1,ll2]=modelFreeRL(data(:,4),data(:,5), data(:,8), [0.4 0.4 1 1 1] )

%worst sll is 278 (2*201*log(0.5), 2 choices for each of 201 trials with 0.5 prob)

%two ways to run this;
%A: given the choices that a subject made, how well would the model predict
%the choices P(modchoice_t|subchoice_1:t, stim_1:t)
%B: given the stimuli and options faced by a subject, what choices would
%the model make P(modchoice_t|modchoice_1:t, stim_1:t)
%note that problems where information does not depend on the choices are
%not affected (unlike eg n-armed bandit tasks)
%B allows the model to make the choices
%B is activated if no choice is provided to the model

%Q; Do we need softmax?
%Q: is choice calculated after learning??? IS ORDER WRONG!!!???
%order
%given choice 1, V1 -> prob1, (s1), state 2
%given choice 2, V2 -> prob2, (s2), (r)
%given choices1 &2, r ->V1,V2

ntrials=length(r);

if size(r,2)>1 %rewards is either actual rewards (nx1) or reward probabilities (nx4)
   rewprob=r;
   r=zeros(ntrials,1);
end

if ~exist('V1')
	V1=zeros(ntrials,2)+0.5;
	V2=zeros(ntrials,4)+0.5;
end

alpha1=par(1);
alpha2=par(2);
lambda=par(3);
beta1=par(4);
beta2=par(5);
pi=par(6);

%only updating the chosen
%done in temporal loop
for tidx=1:ntrials
	%assume no change?
	V2(tidx+1,:)=V2(tidx,:);
	V1(tidx+1,:)=V1(tidx,:);
    
    %if no choice is given (let's see what the algorithm does)
    if s1(tidx)==0
        if tidx>1
            %make choice s1 based on learnt, hard function (no softmax???)
            s1(tidx)=1+(prob1(tidx-1,1)>0.5);
            if s1(tidx)==1
                state2=1+(rand(1)<0.7); %1 or 2
            else %if 2 was chosen
                state2=1+(rand(1)<0.3);
            end
            s2(tidx)=1+(prob2(tidx-1,1+state2)>0.5)+2*state2; %3,4, 5 or 6
            %make choice s2 based on learnt and assigned choice (3 vs 4) or (5 vs 6)
            
        else %random first choice
            s1(tidx)=1+rand(1)>0.5;
            if s1(tidx)==1
                state2(tidx)=1+(rand(1)<0.7); %1 or 2
            else %if 2 was chosen
                state2(tidx)=1+(rand(1)<0.3);
            end
            %second choice depends on assigment
            s2(tidx)=1+(rand(1)>0.5)+2*state2; %3,4, 5 or 6
        end
        
        %get reward, note here stochastic
        r(tidx)=rand(1)<rewprob(tidx,s2(tidx)-2);
    end
    
	%calculate potential change for state s1(tidx) and s2(tidx)
	dV2(tidx)=alpha2*(r(tidx)-V2(tidx,s2(tidx)-2));
	V2(tidx+1,s2(tidx)-2)=V2(tidx,s2(tidx)-2)+dV2(tidx);

	dV1(tidx)=alpha1*(V2(tidx,s2(tidx)-2)-V1(tidx,s1(tidx))) + lambda*alpha1*(r(tidx)-V1(tidx,s1(tidx)));
	V1(tidx+1,s1(tidx))=V1(tidx,s1(tidx))+ dV1(tidx);
    
    %calculate probability of choice within loop to create s1 and s2 for
    %next round
    
    %persisten choice for choice 1
    if tidx==1
        persistentChoice1(tidx,:)=[0 0];
    else
        persistentChoice1(tidx,:)=[s1(tidx-1)==1 s1(tidx-1)==2];
    end
    
    %choice 1, between option 1 and 2
    expChoi1(tidx,:)=exp(beta1*V1(tidx,:)+pi*persistentChoice1(tidx,:));
    prob1(tidx,:)=expChoi1(tidx,:)./repmat(sum(expChoi1(tidx,:),2),1,2);
    
    %choice 2 between option 3 and 4, or between 5 and 6 !!!
    %prob2(tidx,:)=exp(beta2*V2(tidx,:))./repmat(sum(exp(beta2*V2(tidx,:)),2),1,4);
    expChoi2(tidx,:)=exp(beta2*V2(tidx,:));
    sexp1=sum(expChoi2(tidx,1:2));
    sexp2=sum(expChoi2(tidx,3:4));
    prob2(tidx,:)=expChoi2(tidx,:)./[ sexp1 sexp1 sexp2 sexp2];
    
end

if 0
%what was previous choice, do we repeat?
%persistentChoice1=[0 0 0 0;s1(1:ntrials-1)];
persistentChoice1=[0 0;s1(1:ntrials-1)==1 s1(1:ntrials-1)==2];

%calculate the probability of choosing each option
expChoi1=exp(beta1*V1(1:ntrials,:)+pi*persistentChoice1);
prob1=expChoi1./repmat(sum(expChoi1,2),1,2);
%prob1=exp(beta1*V1(1:ntrials,:)+pi*persistentChoice1)./repmat(sum(beta1*V1(1:ntrials,:),2),1,);

prob2=exp(beta2*V2(1:ntrials,:))./repmat(sum(exp(beta2*V2(1:ntrials,:)),2),1,4);
end

%calculate likelihood of model P(s1,s2|parameters)
ll1=diag(prob1(:,s1));
ll2=diag(prob2(:,s2-2));
sll=sum(ll1)+sum(ll2);
if nargout>1
figure
plot(rewprob)
figure
plot(prob2)
figure
plot(V2)
figure
plot(V1)
end
