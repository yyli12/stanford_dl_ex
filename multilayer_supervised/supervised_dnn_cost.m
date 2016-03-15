function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

m = size(data, 2);
n = size(data, 1);

act_fun = @sigmoid;

%% forward prop
%%% YOUR CODE HERE %%%
for hl = 1 : numHidden % hl = hidden layer
    if hl == 1
        hAct{hl}.z = stack{hl}.W * data;
    else
        hAct{hl}.z = stack{hl}.W * hAct{hl-1}.a;
    end
    hAct{hl}.z = hAct{hl}.z + repmat(stack{hl}.b, 1, m);
    hAct{hl}.a = act_fun(hAct{hl}.z);
end

% opl = output layer
opl = exp(stack{numHidden+1}.W * hAct{numHidden}.a + repmat(stack{numHidden+1}.b, 1, m));
pred_prob = bsxfun(@rdivide, opl, sum(opl, 1));
hAct{numHidden+1}.a = pred_prob;

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
ground_truth = full(sparse(labels, 1:m, 1));
ceCost = - sum(sum(ground_truth .* log(pred_prob)));

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
idx = sub2ind(size(pred_prob), labels', 1:size(pred_prob,2));  
error = pred_prob - ground_truth;

for l = numHidden+1 : -1 :1
    gradStack{l}.b = sum(error,2);
    if(l == 1)
        gradStack{l}.W = error * data';
        break;
    else
        gradStack{l}.W = error * hAct{l-1}.a';
    end
    error = (stack{l}.W)' * error .* hAct{l-1}.a .* (1 - hAct{l-1}.a);%此处的error对应是l-1层的error
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;  
for l = 1 : numHidden+1  
    wCost = wCost + .5 * ei.lambda * sum(stack{l}.W(:) .^ 2);  
end  
  
cost = ceCost + wCost;  
  
% Computing the gradient of the weight decay.  
for l = numHidden : -1 : 1  
    gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W;
end  

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



