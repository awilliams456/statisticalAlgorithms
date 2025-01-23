prompt = "What is the adjacency matrix? ";
A = input(prompt) %#ok<NOPTS>
prompt2 = "What is the vector of weights? ";
w = input(prompt2) %#ok<NOPTS>
prompt3 = "What is the number of iterations? ";
its = input(prompt3) %#ok<NOPTS>
%%
Z = sumprod(A, w, its) %#ok<NOPTS>

x = maxprod(A, w, its) %#ok<NOPTS>
function Z = sumprod(A, w, its)
    % create digraph
    G = digraph(A);
    k = numel(w);
    G.Nodes.Belief = ones(numnodes(G),k);
    G.Edges.Belief = ones(numedges(G),k);
    G.Edges.SInMessages = ones(numedges(G),k);
    G.Edges.SOutMessages = ones(numedges(G),k);
    G.Edges.TInMessages = ones(numedges(G),k);
    G.Edges.TOutMessages = ones(numedges(G),k);
    

    % iterate through the sum-product algorithm its times
    for iteration = 1:its
        for i = 1:numnodes(G)
            % calculate all messages from i to C
            for c = transpose(outedges(G,findnode(G,i)))
                [a,b] = findedge(G,c);
                incoming = inedges(G,findnode(G,i));
                incoming(find(incoming == findedge(G, b, a))) = []; %#ok<FNDSB>
                if isempty(incoming)
                    G.Edges.TInMessages(c,:) = exp(w);
                else
                    G.Edges.TInMessages(c,:) = exp(w)*prod(G.Edges.SOutMessages(incoming,:))/sum(exp(w)*prod(G.Edges.SOutMessages(incoming, :)));
                end
            end
            % calculate all messages from C to i
            for c = transpose(inedges(G,findnode(G,i)))
                psi = ones(k);
                for pi = 1:k
                    psi(pi,pi) = 0;
                end
                G.Edges.TOutMessages(c,:) = sum(psi*prod(G.Edges.SInMessages(c,:)))/sum(sum(psi*prod(G.Edges.SInMessages(c,:))));
            end
        end
        G.Edges.SInMessages = G.Edges.TInMessages;
        G.Edges.SOutMessages = G.Edges.TOutMessages;
    end
    % defining convergence as the maximum number of iterations being reached
    
    % calculate beliefs
    for i = 1:numnodes(G)
        for color = 1:k
            belief = (exp(w(color))*prod(G.Edges.TInMessages(inedges(G,findnode(G,i)),:)))/(exp(w)*transpose(prod(G.Edges.TInMessages(inedges(G,findnode(G,i)),:)))) %#ok<NOPRT>
            G.Nodes.Belief(i,:) = belief
        end
    end
    for c = 1:numedges(G)
        [a,b] = findedge(G,c);
        c2 = findedge(G,b,a);
        psi = ones(k);
        for pi = 1:k
            psi(pi,pi) = 0;
        end
        messages = [G.Edges.SInMessages(c,:);G.Edges.SInMessages(c2,:)];
        belief = (psi*transpose(prod(messages)))/sum(psi*transpose(prod(messages)))
        G.Edges.Belief(c,:) = belief
    end
    % calculate Z using bethe free energy
    Z = exp(-(sum(G.Nodes.Belief, "all")-sum(G.Edges.Belief,"all")))
end

function x = maxprod(A, w, its)
    % create digraph
    G = digraph(A);
    k = numel(w);
    G.Nodes.Belief = ones(numnodes(G),1);
    G.Edges.Belief = ones(numedges(G),1);
    G.Edges.SInMessages = ones(numedges(G),k);
    G.Edges.SOutMessages = ones(numedges(G),k);
    G.Edges.TInMessages = ones(numedges(G),k);
    G.Edges.TOutMessages = ones(numedges(G),k);
    

    % iterate through the sum-product algorithm its times
    for iteration = 1:its
        for i = 1:numnodes(G)
            % calculate all messages from i to C
            for c = transpose(outedges(G,findnode(G,i)))
                [a,b] = findedge(G,c);
                incoming = inedges(G,findnode(G,i));
                incoming(find(incoming == findedge(G, b, a))) = []; %#ok<FNDSB>
                if isempty(incoming)
                    G.Edges.TInMessages(c,:) = exp(w);
                else
                    G.Edges.TInMessages(c,:) = exp(w)*prod(G.Edges.SOutMessages(incoming,:))/sum(exp(w)*prod(G.Edges.SOutMessages(incoming, :)));
                end
            end
            % calculate all messages from C to i
            for c = transpose(inedges(G,findnode(G,i)))
                psi = ones(k);
                for pi = 1:k
                    psi(pi,pi) = 0;
                end
                G.Edges.TOutMessages(c,:) = max(psi*prod(G.Edges.SInMessages(c,:)))/sum(psi*prod(G.Edges.SInMessages(c,:)));
            end
        end
        G.Edges.SInMessages = G.Edges.TInMessages;
        G.Edges.SOutMessages = G.Edges.TOutMessages;
    end
    % defining convergence as the maximum number of iterations being reached
    
    % calculate beliefs
    for i = 1:numnodes(G)
        for color = 1:k
            
        end
    end
    for c = 1:numedges(G)
        [a,b] = findedge(G,c);
        c2 = findedge(G,b,a);
        psi = ones(k);
        for pi = 1:k
            psi(pi,pi) = 0;
        end
        messages = [G.Edges.SInMessages(c,:);G.Edges.SInMessages(c2,:)];
        
    end
    x = {}
    for i = 1:numnodes(G)
        maxb = max(G.Nodes.Belief(i,:))
        find(G.Nodes.Belief(i,:) == maxb)
        if size(find(G.Nodes.Belief(i,:) == maxb),2) > 1
            x(:,i) = 0
        else
            
        end
    end
end