function [m,samples] = gibbs(A, w, burnin, its)
% This function uses the Gibbs sampling algorithm to approximate the
% marginals of the probability distribution over weighted vertex
% colorings
%%
% initialize important vars
n = size(A,1); % n is the number of variables in the input graph
Values = 1:length(w); % Values is the list of possible colors
k = length(Values); % k is the number of colors
m = zeros([n k]); % s is the preallocated array of counts for sample values
% for each variable, which will be changed to probabilities with
% normalization
samples = zeros([n its]);
% initialize all nodes
for i=1:n
    node(i).label = i;
    node(i).Value = Values(1);
    node(i).Neighbors = find(A(i,:)~=0);
end
%%
% get all permutations with repetition
C = cell(n,1);
[C{:}] = ndgrid(Values);
y = cellfun(@(Values){Values(:)}, C);
y = [y{:}];

% identify the first valid permutation and set the values of the vectors to
% the first valid coloring found
i = 1;
assignments = size(y,1);
assignment_found = false;
while i <= assignments && assignment_found == false
    conflict = false;
    j = 1;
    while j <= n && conflict == false
        neighbors = node(j).Neighbors;
        for neighbor = neighbors
            if y(i,neighbor) == y(i,j)
                conflict = true;
            end
        end
        j = j+1;
    end
    if conflict == false
        assignment_found = true;
        for j=1:n
            node(j).Value = Values(y(i,j));
        end
    else
        i = i + 1;
    end
end
%%
% loop through the number of times for burnin and then the number of times
% for its
total_its = burnin + its;
for t=1:total_its
    conditionals = zeros([n k]);
    numerators = zeros([n k]);
    for i = 1:n
        % calculate p(x_i|x_\i) for each possible value of x_i
        for x_i = Values
            neighbor_vals = [node(node(i).Neighbors).Value];
            if ~ismember(x_i, neighbor_vals)
                % set equal to exp w because prod(psi) = 1
                numerators(i, x_i) = exp(w(x_i))/(sum(exp(w)));
            end
        end
        conditionals(i,:) = rdivide(numerators(i,:),sum(numerators(i,:)));
        R = rand;
        cumulative = 0;
        c = 1;
        sampled = false;
        while c <= k && sampled == false
            cumulative = cumulative + conditionals(i,c);
            if R <= cumulative
                sampled = true;
                node(i).Value = Values(c);
                if t > burnin
                    m(i, Values(c)) = m(i, Values(c)) + 1;
                    samples(i, t-burnin) = c;
                end
            end
            c = c + 1;
        end
    end
end
%%
% calculate the marginals based on the samples
m = rdivide(m,its);
end