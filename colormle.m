function [w] = colormle(A,samples)
% This function uses Maximum Likelihood Estimation to determine the weights
% that most likely produced a set of samples.
%%
% initialize important vars
n = size(A,1); % n is the number of nodes in the graph
m = size(samples,2); % m is the number of samples
% get the number of potential colors
[~, ~, s] = find(samples);
k = size(unique(s.'), 2);
% get counts of each color
k_counts = zeros(1,k);
for i=1:k
    k_counts(i) = size(find(s==i),1);
end
% initialize all weights to ones
w = ones(1,k);
%%
mle_found = false;
while mle_found == false
    % get sum over all vertexes i of the probability that the color of i is
    % color k according to belief propagation
    converged = zeros(1,k);
    for kk = 1:k
        if converged(kk) == false
            % change the ~ to z if you also want to output the log
            % likelihood
            [~,marginals] = sumprod(A,w,100);
            % calculate the gradient for this color and w
            gradient = k_counts(kk) - (sum(marginals(:,kk))*m);
            % calculate the current log likelihood
            % this has been commented out to save on computational time
            % since it is not strictly necessary
            % logl = sum(k_counts.*w) - (m*log(z))
    
            % if gradient is not close enough to 0, take a step for this
            % color's weight
            newtheta = w(kk) + (0.00001*gradient);
            if abs(newtheta - w(kk)) <= 0.0001
                converged(kk) = true;
            end
            w(kk) = newtheta;
        end
    end
    % if all values of w have converged, then this should be the final
    % result
    if all(converged)
        mle_found = true;
    end
end

end

% from here on is the sumprod algorithm provided, I have modified it to
% also return the marginals of the individual variables
function [Z, marginals] = sumprod(A,w,its)
% This function works for coloring problem using edges as the cliques.
% For making sure of everything please see lecture slides Approximate MAP
% Inference and Variational Inference.

m = size(A,1); % m is the number of variables in the input graph.
Values = 1:length(w); % possible lables for each random variable
k = length(Values); %number of colors

% Initial Nodes
% Each node, or variable, has to have a list of neighbors and a matrix of
% incoming messages. Each message is a function of the assigned value to
% the message reciver and the lable of the sender. So the matrix of reciving messages is a k by m matrix.
for i = 1:m
    node(i).label = i;
    node(i).Neighbors = find(A(i,:)~=0); 
    node(i).Messages = zeros(k,m); %initial messages matrix.
    node(i).Messages(:,node(i).Neighbors) = 1; % initial messages from neighbors by 1.
end
%Normalize Messages.
%Normalizing is always among incoming messages of each variable from a particular neighbor. So
%For each variable
for i = 1:m
    node(i).Messages = node(i).Messages ./ repmat(sum(node(i).Messages),k,1);
    % This line of code is equal to the below for loop
%     %For each neighbor of i 
%     for j = node(i).Neighbors
%         node(i).Messages(:,j) = node(i).Messages(:,j)./sum(node(i).Messages(:,j));
%     end
end
for i = 1:m
    % In each iteration, we use the previously calculated messages
    % to calculate new ones, then we upgrade all the messages at once.
    node(i).MessagesOld = node(i).Messages;
end
%%
%Start Iterations
for it = 1:its
    %For each variable
    for i = 1:m
        %For each possible value of variable i
        for x_i = Values
            %For each each neighbor of i
            for j = node(i).Neighbors
                mySum = 0;
                %For each possible value of j which is not equal to the
                %assignen value of i (because psi(x_i,x_i) = 0)
                % Here, instead of having psi(x_i,x_i) = 0, we just remove
                % x_j = x_i from our calculation using setdiff
                for x_j = setdiff(Values,x_i)
                    %Pick all the neighbors of j, but i
                    NsOfJ = setdiff(node(j).Neighbors,i);
                    nProd = 1;
                    %Get the product of income messages to j
                    for nei = NsOfJ
                        nProd = nProd*node(j).MessagesOld(x_j,nei);
                    end
                    %multiply the resulted product to the phi(x_j)
                    mySum = mySum + phi(w(x_j))*nProd*(x_i~=x_j);
                end
                %Set the income message of node i from j by having x_i as assignment
                node(i).Messages(x_i,j) = mySum;
            end
        end
    end
    %Normalize Messages in each iteration
    for i = 1:m
        node(i).Messages = node(i).Messages ./ repmat(sum(node(i).Messages),k,1);
        node(i).MessagesOld = node(i).Messages;
    end
end

%%
%Calculating believes of variables
% For each variable
% See the details on lecture slide Approximate MAP Inference
for i = 1:m
    % For each possible value of variable i
    for x_i = Values
        node(i).Belief(x_i) = phi(w(x_i))*prod(node(i).Messages(x_i,node(i).Neighbors));
    end
    node(i).Belief = node(i).Belief./(sum(node(i).Belief));
end

% For each edge (in this code we take edges as our cliques)
for i = 1:m
    for j = node(i).Neighbors
        for x_i = Values
            for x_j = setdiff(Values,x_i)
                % These 4 fors are assigning all possible assignments to
                % all the existing edges.
                % The belief of each clique is a function of its assignmnet
                edge(i,j).Belief(x_i,x_j) = phi(w(x_i))*phi(w(x_j))*...
                    prod(node(i).Messages(x_i,setdiff(node(i).Neighbors,j)))*...
                    prod(node(j).Messages(x_j,setdiff(node(j).Neighbors,i)));
            end
        end
        % Normalizing the beliefs of edge (i,j)
        edge(i,j).Belief = edge(i,j).Belief./(sum(sum(edge(i,j).Belief)));
    end
end

%%
% create the marginals table
marginals = zeros(m,k);
for i=1:m
    marginals(i,:) = node(i).Belief;
end
%%
energy = 0;
entropy = 0;

% For calculating energy, we must calculat b_C(X_C).log(psi_C(X_C)) and
% then sum them up. We are supposed to take all the individual variables
% into account. It means in this particular example, we have to calculate
% b_C(X_C).log(psi_C(X_C)) for all the edges and all the variable. As
% psi_edge is an indicator function and we have to make sure that we are
% not going to end up with log(0), we always have log(1). But for each variable we get:
for i =1:m
    for x_i = Values
        % Variables:
        energy = energy + node(i).Belief(x_i)*log(phi(x_i));
        % Edges:
        for j = setdiff(node(i).Neighbors,1:i)
            for x_j = setdiff(Values,x_i)
                energy  = energy + edge(i,j).Belief(x_i,x_j)*log(double(x_i~=x_j));
            end
        end
    end
end

% For calculating entropy, we also have to consider variables as single
% size cliques (just like energy). So again for each variable and each edge
% we have
for i = 1:m
    for x_i = Values
        entropy = entropy + log((node(i).Belief(x_i))^(node(i).Belief(x_i)));
    end
    for j = setdiff(node(i).Neighbors,1:i)
        for x_i = Values
            for x_j = setdiff(Values,x_i)
                entropy = entropy + ...
                    edge(i,j).Belief(x_i,x_j)...
                    *log(...
                    (edge(i,j).Belief(x_i,x_j))...
                    /...
                    (node(i).Belief(x_i)*node(j).Belief(x_j))...
                    );
            end
        end
    end
end
Z = exp(-entropy +energy);
end



function [out] = phi(px)
out = exp(px);
end

