function [w] = colorem(A, L, samples)
% This function finds the most likely weights based on the EM algorithm.
% inputs:
%   A: nxn adjacency matrix
%   L: n-dimensional binary vector indicating the latent variables
%   samples: nxm k-ary matrix where samples_(i,t) corresponds to the color
%   for vertex i in the t^th sample
% outputs:
%   w: the vector of weights identified by the EM algorithm
%%
n = size(A,1); % n is the number of nodes in the graph
M = size(samples,2); % m is the number of samples
% set latent values to 0
for i = 1:n
    if L(i) == 1
        samples(i,:) = zeros;
    end
end
% get the number of potential colors
[~, ~, s] = find(samples);
K = size(unique(s.'), 2);
% initialize all weights to ones
w = ones(1,K);
% get array of possible colors
Values = 1:K;
% initialize all nodes
for i=1:n
    node(i).label = i;
    node(i).Neighbors = find(A(i,:)~=0);
    node(i).latent = 0;
end
% get all permutations of the values of the latent variables
latent_count = size(L,1);
C = cell(latent_count,1);
[C{:}] = ndgrid(Values);
y = cellfun(@(Values){Values(:)}, C);
H = [y{:}];
H = transpose(H);
% add index to latent vars
l = 0;
for i = 1:n
    if L(i) == 1
        l = l + 1;
        node(i).latent = l;
    end
end
% get the observed counts for each color per sample
k_counts = zeros(M,K);
for i=1:M
    [~, ~, sample] = find(samples(:,i));
    for k=1:K
        k_counts(i,k) = size(find(sample==k),1);
    end
end
% get the latent counts for each potential latent combo
l_counts = zeros(size(H,1),K);
for h=1:size(H,2)
    [~, ~, hidden] = find(H(:,h));
    for k = 1:K
        l_counts(h,k) = size(find(hidden==k),1);
    end
end
% get matrix of qm(xH)
q = zeros(M,size(H,2));
%%
converged = zeros(1,K);
q_converged = zeros(M,size(H,2));
em_found = false;
while em_found == false
    % step one: set qm(xH)
    for m = 1:M
        previousq = q(m,:);
        for h = 1:size(H,2)
            current = H(:,h);
            observed = 1;
            latent = 1;
            oblat = 1;
            twolat = 1;
            for i = 1:n
                if oblat ~= 0 && twolat ~= 0
                    if L(i) == 0
                        observed = observed * exp(w(samples(i,m)));
                        neighbor_vals = [];
                        for neighbor = node(i).Neighbors
                            if node(neighbor).latent ~= 0
                                neighbor_vals(end + 1) = H(node(neighbor).latent,h);
                            end
                        end
                        if ismember(samples(i,m), neighbor_vals)
                            oblat = 0;
                        end
                    else
                        latent = latent * exp(w(current(node(i).latent)));
                        neighbor_vals = [];
                        for neighbor = node(i).Neighbors
                            if node(neighbor).latent ~= 0
                                neighbor_vals(end + 1) = H(node(neighbor).latent,h);
                            end
                        end
                        if ismember(current(node(i).latent), neighbor_vals)
                            twolat = 0;
                        end
                    end
                end
            end
            q(m,h) = observed * latent * oblat * twolat;
        end
        normq = sum(q(m,:));
        for h = 1:size(H,2)
            q(m,h) = q(m,h)/normq;
            if abs(previousq(h) - q(m,h)) <= 0.0001
                q_converged(m,h) = true;
            end  
        end
    end

    % step two run mle for each potential sample and k and sum over the potentials
    for m = 1:M
        for h=1:size(H,2)
            % check that it's a valid coloring before continuing
            valid = true;
            i = 1;
            while i <= n && valid == true
                neighbors = node(i).Neighbors;
                for neighbor = neighbors
                    if node(neighbor).latent == 0 && node(i).latent == 0
                        if samples(i,m) == samples(neighbor,m)
                            valid = false;
                        end
                    elseif node(neighbor).latent == 0 && node(i).latent ~= 0
                        if H(node(i).latent,h) == samples(neighbor,m)
                            valid = false;
                        end
                    elseif node(neighbor).latent ~= 0 && node(i).latent == 0
                        if samples(i,m) == H(node(neighbor).latent,h)
                            valid = false;
                        end
                    else
                        if H(node(i).latent,h) == H(node(neighbor).latent,h)
                            valid = false;
                        end
                    end
                end
                i = i + 1;
            end
            if valid
                for count=1:5
                    for k=1:K
                        if converged(k) == false
                            [~,marginals] = sumprod(A,w,100);
                            gradient = k_counts(m,k) + q(m,h)*l_counts(h,k)-(sum(marginals(:,k))*M);
                            newtheta = w(k) + (0.00001*gradient);
                            if abs(newtheta - w(k)) <= 0.0001
                                converged(k) = true;
                            end
                            w(k) = newtheta;
                        end
                    end
                end
            end
        end
    end
    if all(converged) && all(q_converged)
        em_found = true;
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